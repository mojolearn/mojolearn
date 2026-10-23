# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TREES LANE'S PUBLIC DOOR.

Owned by the `trees` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"<the lane's GPU binding>": "<its host binding>"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level

PASS 1 (2026-09-27): every class here fits its trees through the EXISTING
forest entry points (the rf and gbdt bindings), which carry the
identical contract on the GPU and on their CPU host bindings. The numeric glue
between fits (votes, weights, drops, calibration) runs in `xtrees/ops.mojo`
(rows 160-169); Python keeps only O(estimators) scalars in fixed order.
KernelExplainer's coalition schedule, Bagging's oob R^2 and AdaBoost's
sample_weight normalization are Mojo (lane cgr4-py-compute).
CalibratedClassifierCV's per-class epilogue moved to xtrees on lane
py-misc-prep (strided platt/isotonic apply, `complement_pairs`, native class
columns and 0/1 targets; its Python reference arm is deleted).
"""
import numbers
import weakref

from . import _mojolearn_rf, _mojolearn_x_trees  # noqa: F401  the bindings this door resolves; name NO other (lane_select counts > 3 as a registry)
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, frombytes, full, zeros
from ._buffer import memory_at
from ._labels import is_bool
from ._mode import NumericModeMixin, available_if
from ._class_weights import _trees_x_bind  # noqa: F401  re-exported: callers import it from this door
from ._mode import ClassifierScoreMixin, RegressorScoreMixin
from ._forest_protocol import forest_estimator
from .extratrees import ExtraTreesRegressor, ExtraTreesClassifier as _ET_CLS
from .randomforest import (
    RandomForestClassifier, RandomForestRegressor, _refuse,
    _MODEL_FORMATS as _RF_MODEL_FORMATS, _SAVE_SCHEMA as _RF_SAVE_SCHEMA,
)
from . import _native_archive_bridge, _native_trees_bridge

# The compiled saved-model owner, connected once at import: a tree's `save`
# is one native call.
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled tree-ensemble owner: a DecisionTree*'s `fit`, `predict` and
# `predict_proba` are one native call each (bindings/trees_public_forest.mojo;
# splitter='random' grows one extra-trees tree there).
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "DARTRegressor",
    "DARTClassifier",
    "RandomTreesEmbedding",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "OneVsRestClassifier",
    "CalibratedClassifierCV",
    "TreeExplainer",
    "KernelExplainer",
    "PermutationExplainer",
]


# ----------------------------------------------------------------- helpers
# ----------------------------------------------------------- decision trees
# Reference: scikit-learn `sklearn/tree/_classes.py` (DecisionTreeClassifier
# :716, DecisionTreeRegressor :1100, BaseDecisionTree.fit :230). The learner
# is the RF builder (`ensemble/`, cuML's batched level algorithm) run as ONE
# tree, no bootstrap, every feature: sklearn's CART, with two differences said
# out loud: splits are searched over `n_bins` per-feature quantiles (cuML's
# rule, default 128), not every midpoint; and `splitter='random'` and
# `max_leaf_nodes` (best-first growth) are refused by name.
_DT_DEPTH = None


def _dt_common(splitter, max_features, max_leaf_nodes):
    """The random splitter is ExtraTrees' builder (Geurts: one random
    threshold per candidate feature), fitted as ONE tree without bootstrap;
    best-first growth (`max_leaf_nodes`) exists on that builder only."""
    if splitter not in ("best", "random"):
        raise ValueError(f"splitter must be 'best' or 'random', got {splitter!r}")
    if splitter == "best" and max_leaf_nodes is not None:
        _refuse("max_leaf_nodes with splitter='best'", "best-first growth exists on the random"
                " splitter's builder (extratrees/, DEVIATIONS 466-469); pass splitter='random'.")
    return None if splitter == "random" else max_leaf_nodes


# sample_weight goes through the weighted RF fit entry (the weighted
# objective; its CPU restatement is ensemble/host/rf_oracle.mojo).
@forest_estimator("classifier")
class DecisionTreeClassifier(RandomForestClassifier):
    """A single classification tree (CART).

    Follows scikit-learn's ``DecisionTreeClassifier``: one tree grown on every
    row (no bootstrap) and, by default, every feature. Split thresholds are
    searched over ``n_bins`` per-feature quantile bins (cuML's rule) rather
    than every midpoint, so thresholds can differ from scikit-learn's.
    ``X`` is read as float32; NaN or infinity in ``X`` raises ``ValueError``.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        Split quality measure. ``"log_loss"`` is the same as ``"entropy"``.
    splitter : {"best", "random"}, default="best"
        ``"best"`` searches the quantile bins of every candidate feature.
        ``"random"`` grows one extremely randomized tree (one random threshold
        per candidate feature), the builder of ``ExtraTreesClassifier``.
    max_depth : int or None, default=None
        Maximum depth of the tree. None grows until the other stopping rules
        apply.
    min_samples_split : int, default=2
        Minimum number of rows a node needs before it can be split.
    min_samples_leaf : int, default=1
        Minimum number of rows in each leaf.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the total sample weight (of all rows when no
        ``sample_weight`` is given) required in each leaf, in [0.0, 0.5].
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features considered per split: an int count, a fraction in
        (0, 1], ``"sqrt"``, ``"log2"``, or None for every feature.
    random_state : int or None, default=None
        Seed for the random draws. None means seed 0, so repeated fits give
        the same tree (unlike scikit-learn, where None draws a fresh seed).
    max_leaf_nodes : int or None, default=None
        Grow best-first to at most this many leaves. Supported only with
        ``splitter="random"``; with ``splitter="best"`` it raises
        ``NotImplementedError``.
    min_impurity_decrease : float, default=0.0
        A node is split only if the split decreases impurity by at least this
        much.
    class_weight : dict, "balanced" or None, default=None
        Per-class weights, multiplied into ``sample_weight``.
        ``"balanced_subsample"`` is not supported.
    ccp_alpha : float, default=0.0
        Only 0.0 is accepted; cost-complexity pruning is not implemented and
        any other value raises ``NotImplementedError``.
    monotonic_cst : None, default=None
        Only None is accepted; monotonic constraints raise
        ``NotImplementedError``.
    n_bins : int, default=128
        Number of quantile bins per feature used for the split search.
    device : {"gpu"}, default="gpu"
        Only ``"gpu"`` is accepted. On a CPU-only install the same fit runs
        on the host binding.
    inference_engine : {"auto", "sequential", "parallel_groves"}, default="auto"
        Prediction kernel. ``"auto"`` lets the library choose; in identical
        mode it predicts as ``"sequential"``.
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels seen in ``fit``.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int32)
    >>> model = mojolearn.DecisionTreeClassifier(max_depth=4).fit(X, y)
    >>> proba = model.predict_proba(X)
    """

    def __init__(
        self,
        *,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        monotonic_cst=None,
        n_bins=128,
        device="gpu",
        inference_engine="auto",
    ):
        rf_leaves = _dt_common(splitter, max_features, max_leaf_nodes)
        super().__init__(
            n_estimators=1, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=rf_leaves, min_impurity_decrease=min_impurity_decrease,
            bootstrap=False, random_state=random_state, class_weight=class_weight,
            ccp_alpha=ccp_alpha, monotonic_cst=monotonic_cst, n_bins=n_bins,
            n_streams=1, device=device, inference_engine=inference_engine,
        )
        self.splitter = splitter

    def fit(self, X, y, sample_weight=None):
        """Fit the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes are required.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weights, multiplied into the ``class_weight`` row weights.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, False, None), _TREES_CONTEXT)

    def get_depth(self):
        """Return the depth of the fitted tree."""
        return _trees_depth(self)

    def get_n_leaves(self):
        """Return the number of leaves of the fitted tree."""
        return _trees_n_leaves(self)

    def predict_proba(self, X):
        """Class probabilities: the class distribution of each row's leaf.

        Returns an array of shape (n_samples, n_classes), columns in the
        order of ``classes_``.
        """
        return _TREES(self, "trees_forest_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class label of each row of ``X``."""
        return _TREES(self, "trees_forest_predict", (X,), _TREES_CONTEXT)

    def save(self, path):
        """Save the fitted tree to ``path``.

        Raises ``NotImplementedError`` for a tree fitted with
        ``splitter="random"``.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)


# sample_weight goes through the weighted regressor entry
# (rf_regressor_fit_weighted). n_bins=None resolves at fit through `_dt_tier_bins`.
@forest_estimator("regressor")
class DecisionTreeRegressor(RandomForestRegressor):
    """A single regression tree (CART).

    Follows scikit-learn's ``DecisionTreeRegressor``: one tree grown on every
    row (no bootstrap) and, by default, every feature. Split thresholds are
    searched over per-feature quantile bins (cuML's rule) rather than every
    midpoint, so thresholds can differ from scikit-learn's. ``X`` and ``y``
    are read as float32.

    Parameters
    ----------
    criterion : {"squared_error", "mse", "poisson", "gamma", "inverse_gaussian"}, default="squared_error"
        Split quality measure. ``"mse"`` is the same as ``"squared_error"``.
        ``"absolute_error"`` and ``"friedman_mse"`` raise
        ``NotImplementedError``.
    splitter : {"best", "random"}, default="best"
        ``"best"`` searches the quantile bins of every candidate feature.
        ``"random"`` grows one extremely randomized tree (one random threshold
        per candidate feature), the builder of ``ExtraTreesRegressor``.
    max_depth : int or None, default=None
        Maximum depth of the tree. None grows until the other stopping rules
        apply.
    min_samples_split : int, default=2
        Minimum number of rows a node needs before it can be split.
    min_samples_leaf : int, default=1
        Minimum number of rows in each leaf.
    min_weight_fraction_leaf : float, default=0.0
        Minimum fraction of the total sample weight (of all rows when no
        ``sample_weight`` is given) required in each leaf, in [0.0, 0.5].
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features considered per split: an int count, a fraction in
        (0, 1], ``"sqrt"``, ``"log2"``, or None for every feature.
    random_state : int or None, default=None
        Seed for the random draws. None means seed 0, so repeated fits give
        the same tree (unlike scikit-learn, where None draws a fresh seed).
    max_leaf_nodes : int or None, default=None
        Grow best-first to at most this many leaves. Supported only with
        ``splitter="random"``; with ``splitter="best"`` it raises
        ``NotImplementedError``.
    min_impurity_decrease : float, default=0.0
        A node is split only if the split decreases impurity by at least this
        much.
    ccp_alpha : float, default=0.0
        Only 0.0 is accepted; cost-complexity pruning is not implemented and
        any other value raises ``NotImplementedError``.
    monotonic_cst : None, default=None
        Only None is accepted; monotonic constraints raise
        ``NotImplementedError``.
    n_bins : int or None, default=None
        Number of quantile bins per feature used for the split search. None
        uses the numeric mode's default: 128, or 256 in fast mode.
    device : {"gpu"}, default="gpu"
        Only ``"gpu"`` is accepted. On a CPU-only install the same fit runs
        on the host binding.
    inference_engine : {"auto", "sequential", "parallel_groves"}, default="auto"
        Prediction kernel. ``"auto"`` lets the library choose; in identical
        mode it predicts as ``"sequential"``.
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ rng.random(8, dtype=np.float32)
    >>> model = mojolearn.DecisionTreeRegressor(max_depth=6).fit(X, y)
    >>> pred = model.predict(X)
    """

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        n_bins=None,
        device="gpu",
        inference_engine="auto",
    ):
        rf_leaves = _dt_common(splitter, max_features, max_leaf_nodes)
        # n_bins=None: the tier's default, resolved at fit (`_dt_tier_bins`)
        super().__init__(
            n_estimators=1, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=rf_leaves, min_impurity_decrease=min_impurity_decrease,
            bootstrap=False, random_state=random_state, ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst, n_bins=128 if n_bins is None else n_bins, n_streams=1,
            device=device, inference_engine=inference_engine,
        )
        self.splitter = splitter

    def fit(self, X, y, sample_weight=None):
        """Fit the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weights.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, False, None), _TREES_CONTEXT)

    def get_depth(self):
        """Return the depth of the fitted tree."""
        return _trees_depth(self)

    def get_n_leaves(self):
        """Return the number of leaves of the fitted tree."""
        return _trees_n_leaves(self)

    def predict(self, X):
        """Predict the target of each row of ``X``."""
        return _TREES(self, "trees_forest_predict", (X,), _TREES_CONTEXT)

    def save(self, path):
        """Save the fitted tree to ``path``.

        Raises ``NotImplementedError`` for a tree fitted with
        ``splitter="random"``.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)


# A tree saves in the random forest format, refusing a splitter='random'
# tree first (bindings/archive_schema_native.mojo runs the schema); `load`
# is the random forest's one native call.
_TREE_SAVE_SCHEMA = (
    ("refuse_set", "_random_inner", "NotImplementedError",
     "a splitter='random' tree saves through ExtraTrees; not carried yet"),
    *_RF_SAVE_SCHEMA,
)
_native_archive_bridge.register(DecisionTreeClassifier, _RF_MODEL_FORMATS, _TREE_SAVE_SCHEMA)
_native_archive_bridge.register(DecisionTreeRegressor, _RF_MODEL_FORMATS, _TREE_SAVE_SCHEMA)
_native_trees_bridge.register(DecisionTreeClassifier, "dt_classifier")
_native_trees_bridge.register(DecisionTreeRegressor, "dt_regressor")


def _trees_shape_native(est):
    """[depth, leaf count] of the (first) fitted tree from the x_trees binding
    (lane fam2-forests, `x_trees_tree_shape`). A binary without the export
    (-D MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF) refuses get_depth/get_n_leaves:
    lane py-runtime-b deleted the Python node walk it fell back to."""
    if not hasattr(est, "_offsets"):
        raise RuntimeError("this estimator is not fitted yet")
    try:
        entry = getattr(_trees_x_bind(est), "x_trees_tree_shape", None)  # cpu-route: get_depth/get_n_leaves introspection of the fitted host model arrays, outside fit and predict
    except ImportError:    # a host facade refuses an absent export with ImportError
        entry = None
    if not callable(entry):
        raise NotImplementedError(
            "mojolearn: get_depth/get_n_leaves need the x_trees binding's x_trees_tree_shape "
            "(rebuild without -D MOJOLEARN_IDN_TREE_SHAPE_NATIVE_OFF)")
    lo, hi = int(est._offsets[0]), int(est._offsets[1])
    left = as_i32_c(est._left_child, ndim=1, name="left")[0]
    res = empty((2,), "<i4")
    entry(addr_ro(left, name="left") + 4 * lo, addr(res, name="shape"), [hi - lo])
    return res.tolist()


def _trees_depth(est):
    """Depth of the (first) fitted tree."""
    return int(_trees_shape_native(est)[0])


def _trees_n_leaves(est):
    """Leaf count of the (first) fitted tree."""
    return int(_trees_shape_native(est)[1])


# ------------------------------------------------------------- shared glue
def _trees_seed(random_state):
    if random_state is None:
        return 0
    if is_bool(random_state) or not isinstance(random_state, numbers.Integral) or random_state < 0:
        raise ValueError("random_state must be None or a nonnegative int (a RandomState object is refused:"
                         " draws come from the lane's own counter RNG, xtrees/ops.mojo)")
    return int(random_state)


def _trees_sub_seed(seed, k):
    """The random_state handed to the k-th sub-estimator: a fixed function of
    (seed, k), inside int32."""
    return (seed * 1000003 + 7919 * (k + 1)) % 2147483647


def _trees_clone(est, **overrides):
    """A fresh unfitted copy with the same constructor parameters, plus
    `overrides` for the ones the estimator has (random_state)."""
    params = dict(est.get_params(deep=False)) if hasattr(est, "get_params") else {}
    for k, v in overrides.items():  # glue: estimator overrides by parameter name
        if k in params:
            params[k] = v
    return type(est)(**params)


# lane/apple-fast-trees-ensembles (2026-10-02): the Apple FAST switches of
# this module are BUILD-TIME defines of the x_trees binding (`-D
# MOJOLEARN_TE_<NAME>_OFF` turns one off; xtrees/api_pyapi.mojo `x_trees_fast_switches` is the bit
# set the FAST + Apple build fills, 0 in every other build), read through
# the binding once per fit: no env read. IDENTICAL (the default) runs the
# code above and below them unchanged. Each is ON by default in the FAST +
# Apple build since the M3 A/B (xtrees/api_pyapi.mojo XTREES_FAST_SWITCHES):
#   MOJOLEARN_TE_NATIVE_SPLITS  the cv fold bookkeeping of Stacking /
#     CalibratedClassifierCV, OneVsRest's targets and MultiOutputClassifier's
#     label columns leave their Python row loops (`_trees_native_folds`;
#     the native meta-estimator owner, bindings/trees_public_meta.mojo);
#   MOJOLEARN_TE_ADA_SESSION  the AdaBoost members fit the one staged device
#     copy of X (the exact forest data session DART opens by default);
#   MOJOLEARN_TE_ADA_SESSION_SHARE  with it, later members reuse the first
#     member's quantile tables (may move bits: a quality gate, not a digest).
_TE_NATIVE_SPLITS = 1
_TE_ADA_SESSION = 2
_TE_ADA_SESSION_SHARE = 4
#   MOJOLEARN_KSHAP_FAST_BATCH (lane apple-fast-gap-kapprox2; M3 kernel-shap
#     istella 27,011 -> 15,325 ms, same rel_error): Kernel/Permutation SHAP reuse one host buffer for
#     the synthetic rows across chunks and KernelExplainer solves many rows
#     per launch sweep (`x_trees_kshap_means` + `x_trees_kshap_solve_ey`).
_KSHAP_FAST_BATCH = 8
#   MOJOLEARN_AGN_IDN_SYN_POOL (lane idn-shap-pca; an IDENTICAL build's
#     switch, -D MOJOLEARN_AGN_IDN_SYN_POOL_OFF clears it): Kernel/Permutation
#     SHAP hand every chunk the same host buffer for the synthetic rows (the
#     device side builds them in one pooled buffer). Moves no bit.
_AGN_IDN_SYN_POOL = 16
# TOMBSTONE: MOJOLEARN_KSHAP_FAST_OVERLAP (DROPPED-noise: kernel-shap istella neutral, verdicts batch 6) deleted 2026-10-09
# on lane/owed-deletions-D1 (next chunk's synthetic rows + download overlapped with the model); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_KSHAP_FAST_OVERLAP.patch; record in docs/TOMBSTONES.md.
#   MOJOLEARN_PSHAP_DELTA (FAST Apple default; _OFF rollback): the
#     PermutationExplainer model runs only on the synthetic rows that differ
#     from the previous coalition's (`x_trees_pshap_dsynth` +
#     `x_trees_pshap_dvalues`); see xtrees/agnostic_device.mojo.
_PSHAP_DELTA = 32
#   _XT_IDN_ADA_SESSION (lane fam-forests; an IDENTICAL build's switch, -D
#     MOJOLEARN_IDN_ADA_SESSION_OFF clears it): the AdaBoost members fit the
#     one staged device copy of X through the EXACT session. Moves no bit.
#     Bit 128 since the 2026-10-05 merge (was 32 on the IDENTICAL branch;
#     main took 32 for PSHAP_DELTA; xtrees/api_pyapi.mojo XTREES_FAST_SWITCHES).
_IDN_ADA_SESSION = 128
# TOMBSTONE: MOJOLEARN_PSHAP_FAST_OVERLAP (DROPPED-slower: permutation-shap taxi +26.6%, verdicts batch 6) deleted 2026-10-09
# on lane/owed-deletions-D1 (next chunk's delta rows + download overlapped with the model); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_PSHAP_FAST_OVERLAP.patch; record in docs/TOMBSTONES.md.


def _trees_fast_tier(est):
    """True when `est` runs on the FAST tier (its own `numeric_mode`, else
    the process default)."""
    from . import _backend
    mode = getattr(est, "numeric_mode", None)
    mode = _backend.default_mode() if mode is None else str(mode).strip().lower()
    return mode == "fast"


def _trees_switch(est, bit):
    """True when `est` runs on the FAST tier and its x_trees binding was
    built with the define `bit` stands for (`x_trees_fast_switches`)."""
    if not _trees_fast_tier(est):
        return False
    query = getattr(est._bind(), "x_trees_fast_switches", None)
    return callable(query) and (int(query()) & bit) != 0


def _agn_pool_release(b):
    """Lane idn-all: frees the IDENTICAL explainers' pooled synthetic device
    buffer when an explanation ends (xtrees/agnostic_device.mojo
    `pool_release`), so the pool never outlives the `shap_values` call that
    grew it."""
    try:
        release = getattr(b, "x_trees_agn_pool_release", None)
    except ImportError:    # a host facade refuses an absent export with ImportError
        release = None
    if callable(release):
        release()


def _trees_build_switch(est, bit):
    """True when `est`'s x_trees binding was built with the define `bit`
    stands for, on whichever tier that define belongs to
    (`x_trees_fast_switches`)."""
    query = getattr(est._bind(), "x_trees_fast_switches", None)
    return callable(query) and (int(query()) & bit) != 0


def _trees_py2mojo(est):
    """True when `est`'s x_trees binding runs the wrapper glue (cv folds,
    OneVsRest targets, MultiOutputClassifier columns, binary (1 - p, p)
    rows) itself: `x_trees_py2mojo`, 1 in every build unless `-D
    MOJOLEARN_PY2MOJO_trees_OFF` (lane apple-fast-py2mojo-trees, the A/B
    arm A that restores main's Python row loops)."""
    query = getattr(est._bind(), "x_trees_py2mojo", None)
    return callable(query) and int(query()) != 0


class _TreesEnsembleBase(NumericModeMixin):
    """The lane's binding (`_bind()`) and its row/column helpers."""
    _BINDING = "_mojolearn_x_trees"

    def _arange(self, n):
        """int32 0 .. n - 1, written by the binding (`x_trees_iota_i32`, a
        device kernel on a GPU build), not a Python list of n ints."""
        n = int(n)
        out = empty((n,), "<i4")
        if n > 0:
            self._bind().x_trees_iota_i32(addr(out, name="iota"), [n])
        return out

    def _class_major(self, inits, n):
        """K * n float64, class c's block all `inits[c]` (class-major), filled
        by the binding (`x_trees_fill_class_major_f64`, a device kernel on a
        GPU build); K = 1 is (n,)."""
        K, n = len(inits), int(n)
        vals = Array.from_list([float(v) for v in inits], "<f8")  # glue: the K class inits
        out = empty((K * n,), "<f8")
        if K * n > 0:
            self._bind().x_trees_fill_class_major_f64(addr_ro(vals, name="inits"), addr(out, name="score"), [n, K])
        return out

    def _indices(self, n_pool, n_draw, replace, seed, stream):
        """`n_draw` indices of [0, n_pool): with replacement in draw order;
        without, the drawn SET in ascending index order (xtrees/ops.mojo
        `sample_indices`' random-key selection)."""
        out = empty((n_draw,), "<i4")
        self._bind().x_trees_sample_indices(addr(out, name="indices"),
                                            [int(n_pool), int(n_draw), 1 if replace else 0, int(seed), int(stream)])
        return out

    def _gather(self, Xa, rows, cols):
        n_src, d_src = Xa.shape
        out = empty((len(rows), len(cols)), "<f4")
        self._bind().x_trees_gather_f32(addr_ro(Xa, name="X"), addr_ro(rows, name="rows"), addr_ro(cols, name="cols"),
                                        addr(out, name="gathered"), [n_src, d_src, len(rows), len(cols)])
        return out

    def _gather_codes(self, codes, rows):
        out = empty((len(rows),), "<i4")
        self._bind().x_trees_gather_i32(addr_ro(codes, name="codes"), addr_ro(rows, name="rows"),
                                        addr(out, name="gathered"), [len(codes), len(rows)])
        return out

    def _gather_vec(self, v, rows):
        """A float32 vector gathered at rows (as an n x 1 matrix)."""
        col = Array.from_list([0], "<i4")
        return self._gather(v.reshape((len(v), 1)), rows, col).reshape((len(rows),))

    def _acc_cols(self, acc, x, cols, n, k, weight=1.0):
        xa, _ = as_f32_c(x, ndim=2, name="sub-estimator output")
        self._bind().x_trees_accumulate_cols(addr(acc, name="acc"), addr_ro(xa, name="x"),
                                             addr_ro(cols, name="cols"), [n, k, len(cols), float(weight)])

    def _acc(self, acc, x, n, weight=1.0):
        xa, _ = as_f32_c(x, ndim=1, name="sub-estimator output")
        self._bind().x_trees_accumulate(addr(acc, name="acc"), addr_ro(xa, name="x"), [n, float(weight)])

    def _acc_votes(self, acc, codes, n, k, on, off=0.0):
        self._bind().x_trees_accumulate_onehot(addr(acc, name="acc"), addr_ro(codes, name="codes"),
                                               [n, k, float(on), float(off)])

    def _scale(self, acc, divisor):
        self._bind().x_trees_scale(addr(acc, name="acc"), [len(acc), float(divisor)])

    def _argmax(self, acc, n, k):
        out = empty((n,), "<i4")
        self._bind().x_trees_argmax_rows(addr_ro(acc, name="scores"), addr(out, name="argmax"), [n, k])
        return out

    def _codes_of(self, labels_out, classes):
        """Map an estimator's predicted labels (its classes_ are our codes)
        to int32 codes."""
        return as_i32_c(labels_out, ndim=1, name="predicted codes")[0]


# ------------------------------------------------------------------ Bagging
# Reference: scikit-learn `sklearn/ensemble/_bagging.py` (BaseBagging.fit
# :330, _parallel_build_estimators :120, BaggingClassifier.predict_proba
# :960, BaggingRegressor.predict :1270). Deviations: rows are always drawn by
# INDEXING (sklearn turns bootstrap counts into sample weights when the base
# estimator takes them; the regressor tree here does not); draws come from
# the lane's counter RNG (xtrees/ops.mojo), so `random_state` is an int;
# warm_start is refused by name. oob_score (`_set_oob_score` :1000 / :1310)
# sums each member's output on the rows it never drew, in member order.
class _BaggingBase(_TreesEnsembleBase):
    def __init__(self, estimator, n_estimators, max_samples, max_features, bootstrap,
                 bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # -- lane hr2-gbdt-host: the members as ONE forest fit -------------------
    # A bag of best-splitter decision trees over every feature, without row
    # weights or out-of-bag scoring, IS a random forest of `n_estimators`
    # trees with the member's tree parameters and the bag's bootstrap: the
    # forest builder grows every member on the device in one fit (one
    # staging of X, the trees batched) instead of a Python loop that draws,
    # gathers and stages X per member. The members' rows are drawn by the
    # forest's own bootstrap stream. `estimators_` slices the forest into
    # per-tree members when it is read. `MOJOLEARN_HR2_BAGGING_MEMBERS=1`
    # keeps the per-member loop (the A/B arm; deleted once the gates pass).
    @property
    def estimators_(self):
        """The fitted members, as a list of estimators."""
        if getattr(self, "_estimators", None) is None and getattr(self, "_batched", None) is not None:
            self._estimators = self._slice_members()
        if getattr(self, "_estimators", None) is None:
            raise AttributeError("estimators_")
        return self._estimators

    @estimators_.setter
    def estimators_(self, value):
        self._estimators = value
        self._batched = None

    def _slice_members(self):
        """The batched forest's trees as fitted member estimators."""
        forest, base = self._batched
        offsets = forest._offsets.tolist()  # glue: T + 1 tree boundaries
        no = int(forest._num_outputs)
        # each member's node arrays are native block copies of the forest's
        # (Array slicing), never Python lists of every node
        colid, ques, left, leaves = forest._colid, forest._quesval, forest._left_child, forest._leaves
        seed = _trees_seed(self.random_state)
        out = []
        for t in range(int(forest._n_trees)):  # glue: one member per fitted tree
            lo, hi = offsets[t], offsets[t + 1]
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, t))
            est._refresh_config()
            est._capture_fit_mode()
            est._offsets = Array.from_list([0, hi - lo], "<i4")
            est._colid = colid[lo:hi]
            est._quesval = ques[lo:hi]
            est._left_child = left[lo:hi]
            est._leaves = leaves[lo * no:hi * no]
            est._n_trees = 1
            est._num_outputs = no
            est.n_features_in_ = forest.n_features_in_
            if hasattr(forest, "classes_"):
                est.classes_ = forest.classes_
                est.n_classes_ = forest.n_classes_
            out.append(est)
        return out

class BaggingClassifier(ClassifierScoreMixin, _BaggingBase):
    """A bagging ensemble of classifiers.

    Follows scikit-learn's ``BaggingClassifier``: each member is fitted on a
    random draw of rows and features, and ``predict_proba`` averages the
    members' ``predict_proba`` (a member without ``predict_proba`` votes with
    ``predict``), returned as float64. Any mojolearn classifier can be a
    member; the default is ``DecisionTreeClassifier()``. When the members are
    best-splitter decision trees over every feature, without
    ``sample_weight``, ``oob_score`` or feature bootstrap, the whole bag is
    grown as one random forest fit.

    Parameters
    ----------
    estimator : estimator or None, default=None
        The member to clone. None uses ``DecisionTreeClassifier()``.
        Must be a mojolearn DecisionTree*, RandomForest* or ExtraTrees*
        estimator; any other estimator raises ``NotImplementedError``.
    n_estimators : int, default=10
        Number of members; must be at least 1.
    max_samples : int or float, default=1.0
        Rows drawn for each member: an int count, or a fraction in (0, 1] of
        the rows.
    max_features : int or float, default=1.0
        Features drawn for each member: an int count, or a fraction in (0, 1]
        of the features.
    bootstrap : bool, default=True
        Draw rows with replacement.
    bootstrap_features : bool, default=False
        Draw features with replacement.
    oob_score : bool, default=False
        Score the ensemble on the rows each member did not draw. Requires
        ``bootstrap=True`` (``ValueError`` otherwise).
    warm_start : bool, default=False
        Only False is accepted; True raises ``NotImplementedError``.
    n_jobs : None, default=None
        Only None is accepted; members are fitted one after another on the
        device.
    random_state : int or None, default=None
        Seed for the row and feature draws and the members' seeds. None means
        seed 0, so repeated fits are the same (unlike scikit-learn). A
        ``RandomState`` object raises ``ValueError``.
    verbose : int, default=0
        Only 0 is accepted.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels seen in ``fit``.
    n_classes_ : int
        Number of classes.
    estimators_ : list of estimators
        The fitted members.
    estimators_features_ : list of arrays
        The feature indices each member was fitted on.
    n_features_in_ : int
        Number of features seen in ``fit``.
    oob_score_ : float
        Out-of-bag accuracy (with ``oob_score=True``).
    oob_decision_function_ : array of shape (n_samples, n_classes)
        Out-of-bag class scores, normalized per row (with ``oob_score=True``).
        A row that no member left out gets the uniform 1 / n_classes, where
        scikit-learn would divide 0 by 0.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
    >>> model = mojolearn.BaggingClassifier(n_estimators=20, random_state=0).fit(X, y)
    >>> proba = model.predict_proba(X)
    """
    _estimator_type = "classifier"

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator, n_estimators, max_samples, max_features, bootstrap,
                         bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)

    def fit(self, X, y, sample_weight=None):
        """Fit the bag.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes are required.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weights, passed to each member for the rows it drew. The
            member must accept ``sample_weight``.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Mean of the members' class probabilities, float64.

        Returns an array of shape (n_samples, n_classes), columns in the
        order of ``classes_``.
        """
        return _TREES(self, "trees_boost_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest mean probability."""
        return _TREES(self, "trees_boost_predict", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _TREES(self, "trees_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

class BaggingRegressor(RegressorScoreMixin, _BaggingBase):
    """A bagging ensemble of regressors.

    Follows scikit-learn's ``BaggingRegressor``: each member is fitted on a
    random draw of rows and features, and ``predict`` is the members' mean,
    returned as float64. Any mojolearn regressor can be a member; the default
    is ``DecisionTreeRegressor()``. When the members are best-splitter
    decision trees over every feature, without ``sample_weight``,
    ``oob_score`` or feature bootstrap, the whole bag is grown as one random
    forest fit.

    Parameters
    ----------
    estimator : estimator or None, default=None
        The member to clone. None uses ``DecisionTreeRegressor()``.
        Must be a mojolearn DecisionTree*, RandomForest* or ExtraTrees*
        estimator; any other estimator raises ``NotImplementedError``.
    n_estimators : int, default=10
        Number of members; must be at least 1.
    max_samples : int or float, default=1.0
        Rows drawn for each member: an int count, or a fraction in (0, 1] of
        the rows.
    max_features : int or float, default=1.0
        Features drawn for each member: an int count, or a fraction in (0, 1]
        of the features.
    bootstrap : bool, default=True
        Draw rows with replacement.
    bootstrap_features : bool, default=False
        Draw features with replacement.
    oob_score : bool, default=False
        Score the ensemble on the rows each member did not draw. Requires
        ``bootstrap=True`` (``ValueError`` otherwise).
    warm_start : bool, default=False
        Only False is accepted; True raises ``NotImplementedError``.
    n_jobs : None, default=None
        Only None is accepted; members are fitted one after another on the
        device.
    random_state : int or None, default=None
        Seed for the row and feature draws and the members' seeds. None means
        seed 0, so repeated fits are the same (unlike scikit-learn). A
        ``RandomState`` object raises ``ValueError``.
    verbose : int, default=0
        Only 0 is accepted.

    Attributes
    ----------
    estimators_ : list of estimators
        The fitted members.
    estimators_features_ : list of arrays
        The feature indices each member was fitted on.
    n_features_in_ : int
        Number of features seen in ``fit``.
    oob_score_ : float
        Out-of-bag R^2 (with ``oob_score=True``).
    oob_prediction_ : array of shape (n_samples,)
        Out-of-bag prediction of each row (with ``oob_score=True``).

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ rng.random(8, dtype=np.float32)
    >>> model = mojolearn.BaggingRegressor(n_estimators=20, random_state=0).fit(X, y)
    >>> pred = model.predict(X)
    """
    _estimator_type = "regressor"

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator, n_estimators, max_samples, max_features, bootstrap,
                         bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)

    def fit(self, X, y, sample_weight=None):
        """Fit the bag.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Target values, read as float32.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weights, passed to each member for the rows it drew. The
            member must accept ``sample_weight`` (the default
            ``DecisionTreeRegressor`` does not).

        Returns
        -------
        self
        """
        return _TREES(self, "trees_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Mean of the members' predictions, float64."""
        return _TREES(self, "trees_boost_predict", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2; 1.0 is a perfect fit.
        """
        return _TREES(self, "trees_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

# ----------------------------------------------------------------- AdaBoost
# Reference: scikit-learn `sklearn/ensemble/_weight_boosting.py`
# (BaseWeightBoosting.fit :118, AdaBoostClassifier._boost_discrete :560 --
# SAMME, the only algorithm sklearn 1.6+ keeps --, decision_function :660,
# _compute_proba_from_decision :740; AdaBoostRegressor._boost :1030 --
# AdaBoost.R2 --, _get_median_predict :1120). The weight update, the error,
# the estimator weight and the weighted median are `xtrees/ops.mojo`
# (`samme_step`, `r2_step`, `weighted_median`); R2's weighted bootstrap is the
# lane's counter RNG, not numpy's `choice`.
class _AdaBoostBase(_TreesEnsembleBase):
    def __init__(self, estimator, n_estimators, learning_rate, random_state):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

class AdaBoostClassifier(ClassifierScoreMixin, _AdaBoostBase):
    """An AdaBoost classifier (SAMME).

    Follows scikit-learn's ``AdaBoostClassifier`` with the SAMME algorithm:
    each member is fitted on every row, with the current boosting weights
    (scaled by the number of rows, float32) as its ``sample_weight``. Any
    mojolearn classifier that accepts ``sample_weight`` can be a member; the
    default is a decision stump, ``DecisionTreeClassifier(max_depth=1)``.

    Parameters
    ----------
    estimator : estimator or None, default=None
        The member to clone. None uses
        ``DecisionTreeClassifier(max_depth=1)``.
        Must be a mojolearn DecisionTree*, RandomForest* or ExtraTrees*
        estimator; any other estimator raises ``NotImplementedError``.
    n_estimators : int, default=50
        Maximum number of boosting rounds; must be at least 1. Boosting can
        stop earlier, as in scikit-learn.
    learning_rate : float, default=1.0
        Shrinks each member's weight; must be > 0.
    algorithm : {"SAMME"}, default="SAMME"
        Only SAMME is supported (scikit-learn 1.6 removed SAMME.R);
        ``"SAMME.R"`` raises ``NotImplementedError``.
    random_state : int or None, default=None
        Seed for the members' seeds. None means seed 0, so repeated fits
        are the same (unlike scikit-learn). A ``RandomState`` object raises
        ``ValueError``.

    Attributes
    ----------
    estimators_ : list of estimators
        The fitted members.
    estimator_weights_ : list of float
        Weight of each member.
    estimator_errors_ : list of float
        Weighted training error of each member.
    classes_ : array of shape (n_classes,)
        Class labels seen in ``fit``.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
    >>> model = mojolearn.AdaBoostClassifier(n_estimators=50).fit(X, y)
    >>> proba = model.predict_proba(X)
    """
    _estimator_type = "classifier"

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm="SAMME",
                 random_state=None):
        super().__init__(estimator, n_estimators, learning_rate, random_state)
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        """Fit the boosted ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32. NaN or infinity raises
            ``ValueError``.
        y : array-like of shape (n_samples,)
            Class labels; at least two classes are required.
        sample_weight : array-like of shape (n_samples,), default=None
            Initial row weights; must be finite, nonnegative and have a
            positive total. None weights every row equally.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the first member is worse than random guessing.
        """
        return _TREES(self, "trees_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def decision_function(self, X):
        """Weighted member votes, float64.

        Returns shape (n_samples,) for two classes (positive favors
        ``classes_[1]``) and (n_samples, n_classes) otherwise.
        """
        return _TREES(self, "trees_boost_decision_function", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities from the decision function (softmax), float64.

        Returns an array of shape (n_samples, n_classes), columns in the
        order of ``classes_``.
        """
        return _TREES(self, "trees_boost_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest weighted vote."""
        return _TREES(self, "trees_boost_predict", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _TREES(self, "trees_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

class AdaBoostRegressor(RegressorScoreMixin, _AdaBoostBase):
    """An AdaBoost regressor (AdaBoost.R2).

    Follows scikit-learn's ``AdaBoostRegressor``: each member is fitted on a
    weighted bootstrap draw of the rows, and ``predict`` is the weighted
    median of the members' predictions. Any mojolearn regressor can be a
    member; the default is ``DecisionTreeRegressor(max_depth=3)``. The
    bootstrap draws come from mojolearn's own random generator, so they
    differ from scikit-learn's for the same seed.

    Parameters
    ----------
    estimator : estimator or None, default=None
        The member to clone. None uses
        ``DecisionTreeRegressor(max_depth=3)``.
        Must be a mojolearn DecisionTree*, RandomForest* or ExtraTrees*
        estimator; any other estimator raises ``NotImplementedError``.
    n_estimators : int, default=50
        Maximum number of boosting rounds; must be at least 1. Boosting can
        stop earlier, as in scikit-learn.
    learning_rate : float, default=1.0
        Shrinks each member's weight; must be > 0.
    loss : {"linear", "square", "exponential"}, default="linear"
        Loss used to update the row weights after each round.
    random_state : int or None, default=None
        Seed for the members' seeds and the bootstrap draws. None means seed 0, so repeated fits
        are the same (unlike scikit-learn). A ``RandomState`` object raises
        ``ValueError``.

    Attributes
    ----------
    estimators_ : list of estimators
        The fitted members.
    estimator_weights_ : list of float
        Weight of each member.
    estimator_errors_ : list of float
        Weighted average loss of each member.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ rng.random(8, dtype=np.float32)
    >>> model = mojolearn.AdaBoostRegressor(n_estimators=50, random_state=0).fit(X, y)
    >>> pred = model.predict(X)
    """
    _estimator_type = "regressor"
    _LOSSES = {"linear": 0, "square": 1, "exponential": 2}

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss="linear",
                 random_state=None):
        super().__init__(estimator, n_estimators, learning_rate, random_state)
        self.loss = loss

    def fit(self, X, y, sample_weight=None):
        """Fit the boosted ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, read as float32.
        y : array-like of shape (n_samples,)
            Target values, read as float32.
        sample_weight : array-like of shape (n_samples,), default=None
            Initial row weights; must be finite, nonnegative and have a
            positive total. None weights every row equally.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Weighted median of the members' predictions."""
        return _TREES(self, "trees_boost_predict", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2; 1.0 is a perfect fit.
        """
        return _TREES(self, "trees_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

# --------------------------------------------------------------------- DART
# Reference: LightGBM `src/boosting/dart.hpp` (DART::DroppingTrees,
# ::Normalize, ::TrainOneIter) over `gbdt.cpp`'s boosting loop, with
# `regression_objective.hpp` (L2), `binary_objective.hpp` (logloss, sigmoid
# 1) and `multiclass_objective.hpp` (MulticlassSoftmax: one tree per class
# per iteration, h = K / (K - 1) p (1 - p), start log(class prior)) for the
# gradients. Carried: drop_rate, max_drop, skip_drop, uniform_drop,
# xgboost_dart_mode, drop_seed, the tree weights and their normalisation
# (per iteration: a dropped iteration drops all of its K trees), the leaf
# values of `CalculateSplittedLeafOutput` -ThresholdL1(sum g, reg_alpha) /
# (sum h + reg_lambda) clipped to max_delta_step, per-tree feature sampling
# (colsample_bytree, `col_sampler.hpp` GetCnt: round(d * fraction), at least
# 1) and row bagging (subsample every subsample_freq iterations; the tree and
# its leaf values see the bag, every row's score moves, `bagging.hpp`).
# DEVIATIONS: each tree is the forest builder's (level-order, quantile bins,
# `num_leaves` as cuML's `max_leaves` cap) fitted by squared error to -g,
# where LightGBM grows leaf-wise on the g/h histogram gain (so
# min_child_weight, LightGBM's min_sum_hessian_in_leaf, is refused); the leaf
# VALUES are LightGBM's. Every draw comes from the lane's counter RNG (drops
# seeded by drop_seed; the bag, one Bernoulli(subsample) per row in row order,
# by bagging_seed; the feature subset, a sorted draw without replacement, by
# feature_fraction_seed), not LightGBM's LCG; the average / log-odds /
# log-prior start is a constant outside the trees (never dropped), where
# LightGBM folds it into tree 0's bias.
class _DARTBase(_TreesEnsembleBase):
    _KIND = 0

    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda,
                 max_bin, drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed,
                 random_state, reg_alpha=0.0, max_delta_step=0.0, colsample_bytree=1.0, subsample=1.0,
                 subsample_freq=0, feature_fraction_seed=2, bagging_seed=3):
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if not float(learning_rate) > 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= float(drop_rate) <= 1.0 or not 0.0 <= float(skip_drop) <= 1.0:
            raise ValueError("drop_rate and skip_drop must be in [0, 1]")
        if float(reg_lambda) < 0 or float(reg_alpha) < 0:
            raise ValueError("reg_lambda and reg_alpha must be >= 0")
        if not 0.0 < float(colsample_bytree) <= 1.0 or not 0.0 < float(subsample) <= 1.0:
            raise ValueError("colsample_bytree and subsample must be in (0, 1]")
        if int(subsample_freq) < 0:
            raise ValueError("subsample_freq must be >= 0")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.drop_rate = drop_rate
        self.max_drop = max_drop
        self.skip_drop = skip_drop
        self.xgboost_dart_mode = xgboost_dart_mode
        self.uniform_drop = uniform_drop
        self.drop_seed = drop_seed
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.max_delta_step = max_delta_step
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.feature_fraction_seed = feature_fraction_seed
        self.bagging_seed = bagging_seed
        for sd in (random_state, drop_seed, feature_fraction_seed, bagging_seed):  # glue: validates the four seed arguments
            _trees_seed(sd)

    # -------------------------------------------- lane/apple-fast-dart
    # The boosting round on the device (xtrees/dart_device.mojo): FAST on
    # every GPU vendor (Apple since lane/apple-fast-dart, NVIDIA and AMD since
    # lane cpu2-l5-trees; default unless -D MOJOLEARN_DART_DEVICE_OFF) and, since lane
    # fam2-forests, every IDENTICAL build (GPU kernels and the host twin
    # xtrees/dart_host.mojo, default unless -D MOJOLEARN_IDN_DART_DEVICE_OFF);
    # the only builds that register x_trees_dart_open; lane py-runtime-b deleted
    # main's host loop `_boost_loop`, so the round is required). The score, gradients and leaf values are
    # float32 on the device and the dropped trees come off and go back as
    # one gathered sum per row (the docstring of dart_device.mojo).
    _DART_VALUES_CAP = 1 << 26

class DARTRegressor(RegressorScoreMixin, _DARTBase):
    """Gradient boosting with DART dropout, squared-error loss.

    Follows LightGBM's ``boosting="dart"`` with the L2 objective: in each
    iteration a random subset of the earlier trees is dropped while the new
    tree is fitted, and the trees are then renormalized. ``predict`` returns
    the raw score as float64. ``X`` and ``y`` are read as float32.

    Each tree is grown by the forest builder (level-order, quantile bins) on
    the negative gradient; leaf values follow LightGBM (the L1/L2-regularized
    Newton step, clipped by ``max_delta_step``). The draws (dropout, bagging,
    feature sampling) come from mojolearn's own random generator, not
    LightGBM's, and the starting score is a constant kept outside the trees.
    A fit with ``n_estimators * n_classes * (2 * num_leaves - 1)`` above
    2**26, or ``num_leaves`` above 32768, raises ``NotImplementedError``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations; must be at least 1.
    learning_rate : float, default=0.1
        Shrinkage; must be > 0.
    num_leaves : int, default=31
        Maximum leaves per tree, applied as a cap on level-order growth
        (LightGBM grows leaf-wise), so trees can differ from LightGBM's.
    max_depth : int, default=-1
        Maximum tree depth; -1 or any value <= 0 means no limit.
    min_child_samples : int, default=20
        Minimum number of rows in each leaf.
    reg_lambda : float, default=0.0
        L2 regularization on leaf values; must be >= 0.
    max_bin : int, default=255
        Number of quantile bins per feature used for the split search.
    drop_rate : float, default=0.1
        Fraction of earlier trees dropped each iteration, in [0, 1].
    max_drop : int, default=50
        Maximum number of trees dropped per iteration.
    skip_drop : float, default=0.5
        Probability of skipping the dropout in an iteration, in [0, 1].
    xgboost_dart_mode : bool, default=False
        Use XGBoost's DART normalization.
    uniform_drop : bool, default=False
        Drop trees uniformly instead of in proportion to their weights.
    drop_seed : int, default=4
        Seed for the dropout draws.
    random_state : int or None, default=None
        Seed for the trees. None means seed 0.
    reg_alpha : float, default=0.0
        L1 regularization on leaf values; must be >= 0.
    max_delta_step : float, default=0.0
        Clip on each leaf value; 0 means no clip.
    colsample_bytree : float, default=1.0
        Fraction of features sampled for each tree, in (0, 1].
    subsample : float, default=1.0
        Fraction of rows bagged, in (0, 1]; used when ``subsample_freq > 0``.
    subsample_freq : int, default=0
        Re-draw the row bag every this many iterations; 0 disables bagging.
    feature_fraction_seed : int, default=2
        Seed for the per-tree feature sampling.
    bagging_seed : int, default=3
        Seed for the row bagging.

    Attributes
    ----------
    init_score_ : float or list of float
        Starting score (one per class for more than two classes).
    tree_weights_ : list of float
        Final weight of each tree after DART normalization.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ rng.random(8, dtype=np.float32)
    >>> model = mojolearn.DARTRegressor(n_estimators=50).fit(X, y)
    >>> pred = model.predict(X)
    """
    _estimator_type = "regressor"
    _KIND = 0

    def __init__(self, *, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                 min_child_samples=20, reg_lambda=0.0, max_bin=255, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, xgboost_dart_mode=False, uniform_drop=False, drop_seed=4, random_state=None,
                 reg_alpha=0.0, max_delta_step=0.0, colsample_bytree=1.0, subsample=1.0, subsample_freq=0,
                 feature_fraction_seed=2, bagging_seed=3):
        super().__init__(n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda,
                         max_bin, drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed,
                         random_state, reg_alpha, max_delta_step, colsample_bytree, subsample, subsample_freq,
                         feature_fraction_seed, bagging_seed)

    def fit(self, X, y):
        """Fit the boosted trees on ``X`` (n_samples, n_features) and target ``y``.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_boost_fit", (X, y, None), _TREES_CONTEXT)

    def predict(self, X):
        """Raw boosted score of each row, float64."""
        return _TREES(self, "trees_boost_predict", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2; 1.0 is a perfect fit.
        """
        return _TREES(self, "trees_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

class DARTClassifier(ClassifierScoreMixin, _DARTBase):
    """Gradient boosting with DART dropout, classification.

    Follows LightGBM's ``boosting="dart"``: binary log loss for two classes,
    multiclass softmax (one tree per class per iteration) for more. In each
    iteration a random subset of the earlier iterations is dropped while the
    new trees are fitted, and the trees are then renormalized. ``X`` is read
    as float32.

    Each tree is grown by the forest builder (level-order, quantile bins) on
    the negative gradient; leaf values follow LightGBM (the L1/L2-regularized
    Newton step, clipped by ``max_delta_step``). The draws (dropout, bagging,
    feature sampling) come from mojolearn's own random generator, not
    LightGBM's, and the starting score is a constant kept outside the trees.
    A fit with ``n_estimators * n_classes * (2 * num_leaves - 1)`` above
    2**26, or ``num_leaves`` above 32768, raises ``NotImplementedError``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations; must be at least 1.
    learning_rate : float, default=0.1
        Shrinkage; must be > 0.
    num_leaves : int, default=31
        Maximum leaves per tree, applied as a cap on level-order growth
        (LightGBM grows leaf-wise), so trees can differ from LightGBM's.
    max_depth : int, default=-1
        Maximum tree depth; -1 or any value <= 0 means no limit.
    min_child_samples : int, default=20
        Minimum number of rows in each leaf.
    reg_lambda : float, default=0.0
        L2 regularization on leaf values; must be >= 0.
    max_bin : int, default=255
        Number of quantile bins per feature used for the split search.
    drop_rate : float, default=0.1
        Fraction of earlier trees dropped each iteration, in [0, 1].
    max_drop : int, default=50
        Maximum number of trees dropped per iteration.
    skip_drop : float, default=0.5
        Probability of skipping the dropout in an iteration, in [0, 1].
    xgboost_dart_mode : bool, default=False
        Use XGBoost's DART normalization.
    uniform_drop : bool, default=False
        Drop trees uniformly instead of in proportion to their weights.
    drop_seed : int, default=4
        Seed for the dropout draws.
    random_state : int or None, default=None
        Seed for the trees. None means seed 0.
    reg_alpha : float, default=0.0
        L1 regularization on leaf values; must be >= 0.
    max_delta_step : float, default=0.0
        Clip on each leaf value; 0 means no clip.
    colsample_bytree : float, default=1.0
        Fraction of features sampled for each tree, in (0, 1].
    subsample : float, default=1.0
        Fraction of rows bagged, in (0, 1]; used when ``subsample_freq > 0``.
    subsample_freq : int, default=0
        Re-draw the row bag every this many iterations; 0 disables bagging.
    feature_fraction_seed : int, default=2
        Seed for the per-tree feature sampling.
    bagging_seed : int, default=3
        Seed for the row bagging.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels seen in ``fit``.
    init_score_ : float or list of float
        Starting score (one per class for more than two classes).
    tree_weights_ : list of float
        Final weight of each tree after DART normalization.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
    >>> model = mojolearn.DARTClassifier(n_estimators=50).fit(X, y)
    >>> proba = model.predict_proba(X)
    """
    _estimator_type = "classifier"
    _KIND = 1

    def __init__(self, *, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                 min_child_samples=20, reg_lambda=0.0, max_bin=255, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, xgboost_dart_mode=False, uniform_drop=False, drop_seed=4, random_state=None,
                 reg_alpha=0.0, max_delta_step=0.0, colsample_bytree=1.0, subsample=1.0, subsample_freq=0,
                 feature_fraction_seed=2, bagging_seed=3):
        super().__init__(n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda,
                         max_bin, drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed,
                         random_state, reg_alpha, max_delta_step, colsample_bytree, subsample, subsample_freq,
                         feature_fraction_seed, bagging_seed)

    def fit(self, X, y):
        """Fit the boosted trees on ``X`` (n_samples, n_features) and labels ``y``.

        At least two classes are required.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_boost_fit", (X, y, None), _TREES_CONTEXT)

    def decision_function(self, X):
        """Raw boosted scores, float64.

        Shape (n_samples,) for two classes (the log-odds of ``classes_[1]``),
        (n_samples, n_classes) otherwise.
        """
        return _TREES(self, "trees_boost_decision_function", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities, float64, shape (n_samples, n_classes).

        For two classes the columns are ``[1 - p, p]`` with ``p`` the sigmoid
        of the raw score; for more, the softmax of the per-class raw scores.
        """
        return _TREES(self, "trees_boost_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest score."""
        return _TREES(self, "trees_boost_predict", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _TREES(self, "trees_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

# ----------------------------------------------------- RandomTreesEmbedding
# Reference: scikit-learn `sklearn/ensemble/_forest.py` RandomTreesEmbedding
# (:2700; fit :2870 draws y ~ U(0, 1) and fits ExtraTreeRegressor members
# with max_features=1; transform :2960 one-hot encodes `apply`, the leaves
# of each tree in node order). The members are this library's
# ExtraTreesRegressor (one fit of n_estimators trees); y is drawn from the
# lane's counter RNG. DEVIATION: the output is a dense float64 Array;
# `sparse_output=True` (sklearn's default) is refused by name, there being no
# sparse container in the NumPy-free layer.
class RandomTreesEmbedding(_TreesEnsembleBase):
    """An unsupervised embedding by totally random trees.

    Follows scikit-learn's ``RandomTreesEmbedding``: a forest of extremely
    randomized trees (one random feature per split) is fitted to uniform
    random targets, and ``transform`` one-hot encodes the leaf each row
    reaches in every tree. Unlike scikit-learn, the output is a dense float64
    array; ``sparse_output=True`` raises ``NotImplementedError``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int, default=5
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of rows a node needs before it can be split.
    min_samples_leaf : int, default=1
        Minimum number of rows in each leaf.
    min_weight_fraction_leaf : float, default=0.0
        Stored for scikit-learn compatibility; not used by the fit.
    max_leaf_nodes : int or None, default=None
        Grow each tree best-first to at most this many leaves.
    min_impurity_decrease : float, default=0.0
        A node is split only if the split decreases impurity by at least this
        much.
    sparse_output : bool, default=False
        Only False is accepted.
    n_jobs : None, default=None
        Only None is accepted.
    random_state : int or None, default=None
        Seed for the random targets and the trees. None means seed 0. A
        ``RandomState`` object raises ``ValueError``.
    verbose : int, default=0
        Only 0 is accepted.
    warm_start : bool, default=False
        Only False is accepted.

    Attributes
    ----------
    forest_ : ExtraTreesRegressor
        The fitted forest.
    n_trees_ : int
        Number of trees.
    ``n_output_features_`` : int
        Width of the embedding: the total number of leaves.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> emb = mojolearn.RandomTreesEmbedding(n_estimators=10, random_state=0).fit_transform(X)
    """

    def __init__(self, n_estimators=100, *, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 sparse_output=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

    # sk-edges: no `sample_weight` parameter. The totally random trees here
    # take no row weights (scikit-learn drops zero-weight rows from every node
    # and weighs the leaf limits), so an advertised parameter could only
    # refuse; it is left out until the builder takes weights.
    def fit(self, X, y=None):
        """Fit the random trees on ``X`` (n_samples, n_features).

        ``y`` is ignored.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_boost_fit", (X,), _TREES_CONTEXT)

    def apply(self, X):
        """Leaf reached by each row in each tree.

        Returns an int32 array of shape (n_samples, n_trees) of node ids,
        numbered within each tree.
        """
        if not hasattr(self, "forest_"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        if d != self.n_features_in_:
            raise ValueError(f"X has {d} features, fit saw {self.n_features_in_}")
        f = self.forest_
        out = empty((n * self.n_trees_,), "<i4")
        self._bind().x_trees_apply(addr_ro(f._offsets, name="offsets"), addr_ro(f._colid, name="colid"),
                                   addr_ro(f._quesval, name="quesval"), addr_ro(f._left_child, name="left"),
                                   addr_ro(Xa, name="X"), addr(out, name="nodes"), [n, d, 0, self.n_trees_])
        return out.reshape((n, self.n_trees_))

    def transform(self, X):
        """One-hot encode the leaves of ``X``.

        Returns a dense float64 array of shape
        (n_samples, ``n_output_features_``).
        """
        return _TREES(self, "trees_boost_transform", (X,), _TREES_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit on ``X``, then return ``transform(X)``."""
        return _TREES(self, "trees_boost_fit_transform", (X,), _TREES_CONTEXT)

# The bagged and boosted ensembles' native owner (bindings/trees_public_boost.mojo):
# each class's schema, the member classes it assembles and their fixed tables.
_BOOST_MEMBERS = dict(trees_dt_classifier=DecisionTreeClassifier, trees_dt_regressor=DecisionTreeRegressor,
                      trees_rf_classifier=RandomForestClassifier, trees_rf_regressor=RandomForestRegressor,
                      trees_ada_losses=AdaBoostRegressor._LOSSES)
_native_trees_bridge.register_boost(BaggingClassifier, "bagging_classifier", **_BOOST_MEMBERS)
_native_trees_bridge.register_boost(BaggingRegressor, "bagging_regressor")
_native_trees_bridge.register_boost(AdaBoostClassifier, "ada_classifier")
_native_trees_bridge.register_boost(AdaBoostRegressor, "ada_regressor")
_native_trees_bridge.register_boost(DARTClassifier, "dart_classifier")
_native_trees_bridge.register_boost(DARTRegressor, "dart_regressor")
_native_trees_bridge.register_boost(RandomTreesEmbedding, "rte")


# ------------------------------------------------------ wrappers: helpers


class _FoldRows(list):
    """The [(train rows, test rows)] of `_trees_native_folds`: int32 views into
    the one buffer the device filled, kept alive here."""
    __slots__ = ("_owner",)


# The device fold law (sklearn's unshuffled StratifiedKFold / KFold) the
# meta-estimator owner launches (bindings/trees_public_meta.mojo
# `_native_folds`), kept here as the reference its test reads.
def _trees_native_folds(est, n_splits, n, codes, n_classes=0):
    """`_trees_splits` for an int cv with the fold assignment and the fold
    row lists made on the device (xtrees/folds_device.mojo through
    `x_trees_device_folds`: sklearn's unshuffled StratifiedKFold / KFold law,
    sklearn's unshuffled StratifiedKFold / KFold law; the host column runs
    the same law in Mojo (lane cgr4-py-compute), every tier
    (lane/apple-fast-trees-ensembles, 2026-10-02). `codes` is an int32 code
    Array in [0, n_classes) (stratified) or None (KFold). The lists are
    zero-copy int32 views into one downloaded buffer: fold i's rows outside
    it, ascending, then inside it, ascending."""
    b = est._bind()
    counts = empty((n_splits + 1,), "<i4")
    rows = empty((n_splits * n,), "<i4")
    if codes is not None:
        c32 = as_i32_c(codes, ndim=1, name="codes")[0]
        k = int(n_classes)
    else:
        c32, k = counts, 0
    status = int(b.x_trees_device_folds(addr_ro(c32, name="codes"), addr(rows, name="rows"),
                                        addr(counts, name="counts"), [n, n_splits, k]))
    if status == 1:
        raise ValueError(f"n_splits={n_splits} cannot be greater than the number of members in each class")
    if status != 0:
        raise ValueError("y codes outside [0, n_classes)")
    cnt = [int(v) for v in counts.tolist()[:n_splits]]  # glue: fold sizes and offsets over n_splits folds
    used = max([i for i in range(n_splits) if cnt[i] > 0] or [0]) + 1  # glue: fold sizes and offsets over n_splits folds
    out = _FoldRows()
    out._owner = rows
    for i in range(used):  # glue: fold sizes and offsets over n_splits folds
        base = rows._addr + 4 * i * n
        tr = memory_at(base, 4 * (n - cnt[i]), writable=False).cast("i")
        te = memory_at(base + 4 * (n - cnt[i]), 4 * cnt[i], writable=False).cast("i")
        out.append((tr, te))
    return out


def _trees_output_2d(x, n):
    """A member's output as a float32 (n, c) Array (a 1-D output is c = 1)."""
    a, _ = as_f32_c(x, ndim=2, name="output") if len(getattr(x, "shape", ())) == 2 else \
        (as_f32_c(x, ndim=1, name="output")[0].reshape((n, 1)), None)
    return a


class _FitTransformMixin:
    """scikit-learn's `TransformerMixin.fit_transform` for the ensembles
    with a `transform` (voting and stacking)."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Targets.
        **fit_params
            Passed to ``fit`` (``sample_weight``).

        Returns
        -------
        array of shape (n_samples, n_output_features), float64
            The transformed training data.
        """
        return _TREES(self, "trees_meta_fit_transform", (X, y, fit_params), _TREES_CONTEXT)


class _MetaClassifierScore:
    """The meta-classifiers' `score`: one native call (the native predict
    and the native accuracy)."""

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        return _TREES(self, "trees_meta_score", (X, y, sample_weight), _TREES_CONTEXT)


class _MetaRegressorScore:
    """The meta-regressors' `score`: one native call (the native predict
    and the native R^2)."""

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _TREES(self, "trees_meta_score", (X, y, sample_weight), _TREES_CONTEXT)


class _TreesWrapperBase(_TreesEnsembleBase):
    """The meta-estimators' base: each public method is one call into
    bindings/trees_public_meta.mojo (operations `trees_meta_*`), which
    fits and queries the members through their own public methods."""


# ------------------------------------------------------------------ Voting
# Reference: scikit-learn `sklearn/ensemble/_voting.py` (VotingClassifier
# :200 -- fit on the label-encoded y, 'hard' = weighted bincount argmax,
# 'soft' = weighted np.average of predict_proba --, VotingRegressor :500 --
# weighted np.average of predict). Members are cloned, never refitted in
# place; n_jobs and verbose are refused by name.
class VotingClassifier(_MetaClassifierScore, ClassifierScoreMixin, _FitTransformMixin, _TreesWrapperBase):
    """Combine classifiers by a weighted majority vote or averaged probabilities.

    Follows scikit-learn's ``VotingClassifier``. ``fit`` clones every estimator
    and fits the clones on the label-encoded ``y``; the estimators you pass are
    never fitted in place. ``voting="hard"`` predicts the weighted majority of
    the members' predicted labels; ``voting="soft"`` predicts the argmax of the
    weighted average of their ``predict_proba``. The members fit one after
    another.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Named mojolearn classifiers. Names must be unique; an entry whose
        estimator is the string ``"drop"`` is skipped. An empty list, duplicate
        names, or every entry ``"drop"`` raise ``ValueError``.
        A member that is not a mojolearn estimator raises ``NotImplementedError``.
    voting : {"hard", "soft"}, default "hard"
        Majority vote on predicted labels, or argmax of the averaged
        probabilities (every member then needs ``predict_proba``). Any other
        value raises ``ValueError``.
    weights : sequence of float, default None
        One weight per active estimator; None weighs them equally. A length
        that does not match the active estimators raises ``ValueError``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    flatten_transform : bool, default True
        Kept for scikit-learn compatibility; ``transform`` always returns a 2-D
        array.
    verbose : bool, default False
        Only False is accepted; True raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        The fitted clones, in order (fitted on integer class codes).
    named_estimators_ : dict
        The fitted clones by name.
    classes_ : array
        The class labels seen in ``fit``, sorted.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int32)
    >>> model = mojolearn.VotingClassifier(
    ...     [("rf", mojolearn.RandomForestClassifier()),
    ...      ("dt", mojolearn.DecisionTreeClassifier())], voting="soft").fit(X, y)
    >>> model.predict(X[:5]).shape
    (5,)
    """
    _estimator_type = "classifier"

    def __init__(self, estimators, *, voting="hard", weights=None, n_jobs=None, flatten_transform=True,
                 verbose=False):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a clone of every active estimator on ``X`` and ``y``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples,)
            Class labels.
        sample_weight : array of shape (n_samples,), default None
            Passed to every member's ``fit`` when given.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    @available_if(lambda self: self.voting != "hard")
    def predict_proba(self, X):
        """Weighted average of the members' class probabilities.

        Only available with ``voting="soft"``; with ``voting="hard"`` it raises
        ``AttributeError``.

        Returns
        -------
        array of shape (n_samples, n_classes), float64
        """
        return _TREES(self, "trees_meta_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict class labels by weighted vote (hard) or averaged probabilities (soft).

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)

    def transform(self, X):
        """The members' outputs side by side.

        Returns
        -------
        array, float64
            With ``voting="soft"``, every member's class probabilities, shape
            (n_samples, n_estimators * n_classes); with ``voting="hard"``, every
            member's predicted class code, shape (n_samples, n_estimators).
        """
        return _TREES(self, "trees_meta_transform", (X,), _TREES_CONTEXT)


class VotingRegressor(_MetaRegressorScore, RegressorScoreMixin, _FitTransformMixin, _TreesWrapperBase):
    """Average the predictions of several regressors.

    Follows scikit-learn's ``VotingRegressor``. ``fit`` clones every estimator
    and fits the clones; the estimators you pass are never fitted in place.
    ``predict`` is the weighted average of the members' predictions. The
    members fit one after another.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Named mojolearn regressors. Names must be unique; an entry whose
        estimator is the string ``"drop"`` is skipped. An empty list, duplicate
        names, or every entry ``"drop"`` raise ``ValueError``.
        A member that is not a mojolearn estimator raises ``NotImplementedError``.
    weights : sequence of float, default None
        One weight per active estimator; None weighs them equally. A length
        that does not match the active estimators raises ``ValueError`` in
        ``predict``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    verbose : bool, default False
        Only False is accepted; True raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        The fitted clones, in order.
    named_estimators_ : dict
        The fitted clones by name.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.VotingRegressor(
    ...     [("rf", mojolearn.RandomForestRegressor()),
    ...      ("dt", mojolearn.DecisionTreeRegressor())]).fit(X, y)
    >>> model.predict(X[:5]).shape
    (5,)
    """
    _estimator_type = "regressor"

    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit a clone of every active estimator on ``X`` and ``y``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples,)
            Targets.
        sample_weight : array of shape (n_samples,), default None
            Passed to every member's ``fit`` when given.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Weighted average of the members' predictions.

        Returns
        -------
        array of shape (n_samples,), float64
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)

    def transform(self, X):
        """Every member's prediction side by side.

        Returns
        -------
        array of shape (n_samples, n_estimators), float64
        """
        return _TREES(self, "trees_meta_transform", (X,), _TREES_CONTEXT)


# ---------------------------------------------------------------- Stacking
# Reference: scikit-learn `sklearn/ensemble/_stacking.py` (_BaseStacking.fit
# :150 -- every member refitted on all rows, the meta features from
# cross_val_predict --, _concatenate_predictions :80 -- a binary
# predict_proba keeps only its second column --, stack_method 'auto' =
# predict_proba, decision_function, predict). cv is an int of unshuffled
# folds (StratifiedKFold for the classifier, KFold for the regressor), a
# splitter object or an iterable of (train, test) pairs whose test sets
# partition the rows (`_trees_splits`).
# DEVIATION: the default final estimators are this library's
# LogisticRegression and Ridge (sklearn: LogisticRegression, RidgeCV).
class _StackingBase(_FitTransformMixin, _TreesWrapperBase):

    def __init__(self, estimators, final_estimator, cv, stack_method, n_jobs, passthrough, verbose):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    def transform(self, X):
        """The meta features the final estimator reads.

        Each member's output (per ``stack_method_``) side by side, followed by
        ``X`` itself when ``passthrough=True``. For a binary classifier,
        ``predict_proba`` contributes only its second column.

        Returns
        -------
        array of shape (n_samples, n_meta_features), float64
        """
        return _TREES(self, "trees_meta_transform", (X,), _TREES_CONTEXT)


class StackingClassifier(_MetaClassifierScore, ClassifierScoreMixin, _StackingBase):
    """Stack classifiers and train a final classifier on their cross-validated outputs.

    Follows scikit-learn's ``StackingClassifier``. ``fit`` refits every member on
    all rows, and trains ``final_estimator`` on the members' out-of-fold
    outputs (``cross_val_predict``). The members and folds fit one after
    another.

    Differs from scikit-learn: the default ``final_estimator`` is
    mojolearn's ``LogisticRegression``.

    Parameters
    ----------
    estimators : list of (str, estimator)
        Named mojolearn classifiers. Names must be unique; an entry whose
        estimator is the string ``"drop"`` is skipped.
        A member that is not a mojolearn estimator raises ``NotImplementedError``.
    final_estimator : estimator, default None
        The classifier fitted on the meta features; None uses
        ``LogisticRegression()``.
        Must be a mojolearn estimator; any other estimator raises ``NotImplementedError``.
    cv : int, splitter or iterable of (train, test), default None
        An int is the number of unshuffled stratified folds (``>= 2``); None
        means 5. A splitter object's ``split(X, y)`` or an iterable of index
        pairs is used as given, but its test sets must cover every row exactly
        once, else ``ValueError``.
    stack_method : {"auto", "predict_proba", "decision_function", "predict"}, default "auto"
        The member method whose output is stacked. ``"auto"`` takes the first
        of ``predict_proba``, ``decision_function``, ``predict`` the member has.
        For a binary problem ``predict_proba`` contributes only its second
        column.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    passthrough : bool, default False
        Also give the final estimator the original features.
    verbose : int, default 0
        Only 0 is accepted; any other value raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        The members refitted on all rows (fitted on integer class codes).
    named_estimators_ : dict
        The fitted members by name.
    stack_method_ : list of str
        The method stacked for each member.
    final_estimator_ : estimator
        The fitted final classifier.
    classes_ : array
        The class labels seen in ``fit``, sorted.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1.0).astype(np.int32)
    >>> model = mojolearn.StackingClassifier(
    ...     [("rf", mojolearn.RandomForestClassifier()),
    ...      ("dt", mojolearn.DecisionTreeClassifier())]).fit(X, y)
    >>> model.predict_proba(X[:5]).shape
    (5, 2)
    """
    _estimator_type = "classifier"

    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method="auto", n_jobs=None,
                 passthrough=False, verbose=0):
        super().__init__(estimators, final_estimator, cv, stack_method, n_jobs, passthrough, verbose)

    def fit(self, X, y, sample_weight=None):
        """Fit the members, their out-of-fold outputs, and the final estimator.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples,)
            Class labels.
        sample_weight : array of shape (n_samples,), default None
            Passed to every member's ``fit`` (each fold's rows) and to the
            final estimator's; every one of them must accept it.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            When a member's output width differs between folds (a class missing
            from a fold's training rows).
        """
        return _TREES(self, "trees_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Predict class labels with the final estimator on the stacked outputs.

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """The final estimator's class probabilities on the stacked outputs.

        Returns
        -------
        array of shape (n_samples, n_classes), float64
        """
        return _TREES(self, "trees_meta_predict_proba", (X,), _TREES_CONTEXT)


class StackingRegressor(_MetaRegressorScore, RegressorScoreMixin, _StackingBase):
    """Stack regressors and train a final regressor on their cross-validated outputs.

    Follows scikit-learn's ``StackingRegressor``. ``fit`` refits every member on
    all rows, and trains ``final_estimator`` on the members' out-of-fold
    ``predict`` outputs (``cross_val_predict``). The members and folds fit one
    after another.

    Differs from scikit-learn: the default ``final_estimator`` is mojolearn's
    ``Ridge`` (scikit-learn uses ``RidgeCV``).

    Parameters
    ----------
    estimators : list of (str, estimator)
        Named mojolearn regressors. Names must be unique; an entry whose
        estimator is the string ``"drop"`` is skipped.
        A member that is not a mojolearn estimator raises ``NotImplementedError``.
    final_estimator : estimator, default None
        The regressor fitted on the meta features; None uses ``Ridge()``.
        Must be a mojolearn estimator; any other estimator raises ``NotImplementedError``.
    cv : int, splitter or iterable of (train, test), default None
        An int is the number of unshuffled ``KFold`` folds (``>= 2``); None
        means 5. A splitter object's ``split(X, y)`` or an iterable of index
        pairs is used as given, but its test sets must cover every row exactly
        once, else ``ValueError``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    passthrough : bool, default False
        Also give the final estimator the original features.
    verbose : int, default 0
        Only 0 is accepted; any other value raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        The members refitted on all rows.
    named_estimators_ : dict
        The fitted members by name.
    stack_method_ : list of str
        The method stacked for each member (``"predict"``).
    final_estimator_ : estimator
        The fitted final regressor.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.StackingRegressor(
    ...     [("rf", mojolearn.RandomForestRegressor()),
    ...      ("dt", mojolearn.DecisionTreeRegressor())]).fit(X, y)
    >>> model.predict(X[:5]).shape
    (5,)
    """
    _estimator_type = "regressor"
    _binary = False

    def __init__(self, estimators, final_estimator=None, *, cv=None, n_jobs=None, passthrough=False, verbose=0):
        super().__init__(estimators, final_estimator, cv, "auto", n_jobs, passthrough, verbose)

    def fit(self, X, y, sample_weight=None):
        """Fit the members, their out-of-fold predictions, and the final estimator.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples,)
            Targets, converted to float32.
        sample_weight : array of shape (n_samples,), default None
            Passed to every member's ``fit`` (each fold's rows) and to the
            final estimator's; every one of them must accept it.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Predict with the final estimator on the stacked member predictions.

        Returns
        -------
        array of shape (n_samples,)
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)


# ------------------------------------------------------------- MultiOutput
# Reference: scikit-learn `sklearn/multioutput.py` (_MultiOutputEstimator.fit
# :200, one clone per column of Y; predict stacks the columns;
# MultiOutputClassifier.predict_proba :500 returns a list). Y is a numeric
# 2-D buffer; a classifier's labels per column are encoded to codes.
#: lane apple-fast-meta (FAST + Apple default, -D MOJOLEARN_MULTIOUT_RIDGE_OFF turns it off):
#: MultiOutputRegressor(Ridge) fits every target in ONE ridge program
#: (glm/impl/ridge_multi.mojo) when the estimators binding exports
#: ridge_fit_multi; the native owner takes that route (trees_public_meta.mojo
#: `_multioutput_ridge_fit`), every other binding the per-target route.
class MultiOutputRegressor(_MetaRegressorScore, RegressorScoreMixin, _TreesWrapperBase):
    """Fit one regressor per target column.

    Follows scikit-learn's ``MultiOutputRegressor``: ``fit`` clones
    ``estimator`` once per column of ``Y`` and fits each clone on that column;
    ``predict`` stacks their predictions. The columns fit one after another.

    Parameters
    ----------
    estimator : estimator
        A mojolearn regressor; it is cloned, never fitted in place.
        Must be a mojolearn estimator; any other estimator raises ``NotImplementedError``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        One fitted clone per column of ``Y``.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> Y = np.stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]], axis=1)
    >>> model = mojolearn.MultiOutputRegressor(mojolearn.Ridge()).fit(X, Y)
    >>> model.predict(X[:5]).shape
    (5, 2)
    """
    _estimator_type = "regressor"
    # scikit-learn's multi-output-only target tags: its checks pass a 2-D y.
    _sklearn_target_tags = {"single_output": False, "multi_output": True}

    def __init__(self, estimator, *, n_jobs=None):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit one clone of ``estimator`` per column of ``y``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples, n_outputs)
            Numeric targets, converted to float32. A row count that differs from
            ``X`` raises ``ValueError``.
        sample_weight : array of shape (n_samples,), default None
            Passed to every clone's ``fit`` when given.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Every target's prediction, one column per output.

        Returns
        -------
        array of shape (n_samples, n_outputs)
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)


class MultiOutputClassifier(_TreesWrapperBase):
    """Fit one classifier per label column.

    Follows scikit-learn's ``MultiOutputClassifier``: ``fit`` clones
    ``estimator`` once per column of ``Y``, encodes that column's labels and
    fits the clone on them; ``predict`` stacks the decoded predictions and
    ``predict_proba`` returns a list with one probability array per output.
    The columns fit one after another.

    Parameters
    ----------
    estimator : estimator
        A mojolearn classifier; it is cloned, never fitted in place.
        Must be a mojolearn estimator; any other estimator raises ``NotImplementedError``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        One fitted clone per column of ``Y`` (fitted on integer class codes).
    classes_ : list of arrays
        The sorted class labels of each output column.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> Y = np.stack([X[:, 0] > 0.5, X[:, 1] > 0.5], axis=1).astype(np.int64)
    >>> model = mojolearn.MultiOutputClassifier(mojolearn.DecisionTreeClassifier()).fit(X, Y)
    >>> model.predict(X[:5]).shape
    (5, 2)
    """
    _estimator_type = "classifier"
    # scikit-learn's multi-output-only target tags: its checks pass a 2-D y.
    _sklearn_target_tags = {"single_output": False, "multi_output": True}

    def __init__(self, estimator, *, n_jobs=None):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit one clone of ``estimator`` per label column of ``y``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples, n_outputs)
            Class labels, one column per output. A row count that differs from
            ``X`` raises ``ValueError``.
        sample_weight : array of shape (n_samples,), default None
            Passed to every clone's ``fit`` when given.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Predict every output's labels.

        Returns
        -------
        array of shape (n_samples, n_outputs)
            For numeric labels, int64 when every output's labels are integers,
            else float64.
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Every output's class probabilities.

        Returns
        -------
        list of arrays
            One float64 array of shape (n_samples, n_classes_j) per output, its
            columns in the order of ``classes_[j]``.
        """
        return _TREES(self, "trees_meta_predict_proba", (X,), _TREES_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Subset accuracy: the fraction of rows whose every output is
        predicted correctly, as scikit-learn's ``MultiOutputClassifier.score``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples, n_outputs)
            True labels, numeric.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of rows with all outputs correct (weighted when
            ``sample_weight`` is given).
        """
        return _TREES(self, "trees_meta_score", (X, y, sample_weight), _TREES_CONTEXT)


# --------------------------------------------------------------- OneVsRest
# Reference: scikit-learn `sklearn/multiclass.py` OneVsRestClassifier (fit
# :330, one binary clone per class -- a single one for two classes --;
# predict :420 argmax of decision_function, else of predict_proba[:, 1];
# predict_proba :470 each class's positive probability, rows normalised in
# the multiclass case). DEVIATION: a row whose scores sum to 0 is uniform,
# where sklearn divides 0 / 0.
class OneVsRestClassifier(_MetaClassifierScore, ClassifierScoreMixin, _TreesWrapperBase):
    """Multiclass classification with one binary classifier per class.

    Follows scikit-learn's ``OneVsRestClassifier``: ``fit`` trains one clone of
    ``estimator`` per class (that class against the rest), or a single clone
    when there are two classes. ``predict`` takes the class whose clone scores
    highest, using ``decision_function`` when the estimator has one, else the
    positive-class column of ``predict_proba``. The classes fit one after
    another.

    Differs from scikit-learn: in ``predict_proba``, a row whose per-class
    probabilities are all 0 gets uniform probabilities (scikit-learn divides
    0 by 0).

    Parameters
    ----------
    estimator : estimator
        A mojolearn binary classifier; it is cloned, never fitted in place.
        Must be a mojolearn estimator; any other estimator raises ``NotImplementedError``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    verbose : int, default 0
        Only 0 is accepted; any other value raises ``NotImplementedError``.

    Attributes
    ----------
    estimators_ : list
        The fitted clones, one per class (one in total for two classes).
    classes_ : array
        The class labels seen in ``fit``, sorted.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (3 * X[:, 0]).astype(np.int32)
    >>> model = mojolearn.OneVsRestClassifier(mojolearn.LogisticRegression()).fit(X, y)
    >>> model.predict_proba(X[:5]).shape
    (5, 3)
    """
    _estimator_type = "classifier"

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Fit one binary clone per class.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples,)
            Class labels; fewer than two classes raise ``ValueError``.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_meta_fit", (X, y, None), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class whose binary classifier scores highest.

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities.

        Each class's positive-class probability from its clone, with every row
        normalized to sum to 1 when there are more than two classes. For two
        classes the rows are ``(1 - p, p)``.

        Returns
        -------
        array of shape (n_samples, n_classes), float64
        """
        return _TREES(self, "trees_meta_predict_proba", (X,), _TREES_CONTEXT)


# ------------------------------------------------------------- Calibration
# Reference: scikit-learn `sklearn/calibration.py` (CalibratedClassifierCV.fit
# :300 -- ensemble=True: one (member, calibrators) pair per fold; False:
# cross_val_predict scores, one calibrator set, the member refitted on all
# rows --, _fit_calibrator :650 one calibrator per class, OvR; _CalibratedClassifier
# .predict_proba :720 -- binary: [1 - p, p]; multiclass: normalised, a zero
# row uniform --, _sigmoid_calibration :800, IsotonicRegression(out_of_bounds
# ='clip')). Scores are decision_function, else predict_proba (its class-1
# column when binary). DEVIATIONS: Platt's minimiser is Newton with
# backtracking (xtrees/ops.mojo platt_fit) on sklearn's objective, not
# L-BFGS; the default estimator is this library's LinearSVC as sklearn's.
#: CalibratedClassifierCV's per-class epilogue runs in xtrees (strided
#: calibrators, native columns and 0/1 targets); the Python reference route
#: (an [fb] arm no install reached: every x_trees binding carries the strided
#: entries) is deleted (lane apple-fast-py2mojo-trees).
#: The x_prep one-program CalibratedClassifierCV(GaussianNB) fit
#: (MOJOLEARN_CALIB_GNB_FOLDS, FAST + Apple) was assembled by the prep
#: family's Python program builder; the native owner takes the reference
#: route (bindings/trees_public_meta.mojo, header note).

class CalibratedClassifierCV(_MetaClassifierScore, ClassifierScoreMixin, _TreesWrapperBase):
    """Calibrate a classifier's probabilities with cross-validation.

    Follows scikit-learn's ``CalibratedClassifierCV``. The calibrator reads the
    estimator's ``decision_function``, else its ``predict_proba`` (the positive
    column for a binary problem), and fits one calibrator per class
    (one-vs-rest): Platt's sigmoid or isotonic regression (clipped outside the
    fitted range). Multiclass probabilities are normalized per row; a row of
    zeros becomes uniform. The folds fit one after another.

    Differs from scikit-learn: Platt's sigmoid minimizes scikit-learn's
    objective with Newton steps and a backtracking line search instead of
    L-BFGS, so its parameters can differ in the last digits.

    Parameters
    ----------
    estimator : estimator, default None
        The mojolearn classifier to calibrate; it is cloned, never fitted in
        place. None uses ``LinearSVC()``, as scikit-learn does.
        Must be a mojolearn estimator; any other estimator raises ``NotImplementedError``.
    method : {"sigmoid", "isotonic"}, default "sigmoid"
        Platt's sigmoid or isotonic regression. Any other value raises
        ``ValueError``.
    cv : int, splitter or iterable of (train, test), default None
        An int is the number of unshuffled stratified folds (``>= 2``); None
        means 5. A splitter object's ``split(X, y)`` or an iterable of index
        pairs is used as given; with ``ensemble=False`` their test sets must
        cover every row exactly once, else ``ValueError``.
    n_jobs : None
        Only None is accepted; any other value raises ``NotImplementedError``.
    ensemble : bool, default True
        True: one (classifier, calibrators) pair per fold, the classifier fitted
        on the fold's training rows and calibrated on its test rows;
        ``predict_proba`` averages the pairs. False: one set of calibrators fitted
        on the out-of-fold scores, and the classifier refitted on all rows.
        Values other than True or False raise ``NotImplementedError``.

    Attributes
    ----------
    calibrated_classifiers_ : list
        One (fitted classifier, calibrators) pair per fold, or a single pair
        when ``ensemble=False``. The classifiers are fitted on integer class
        codes.
    classes_ : array
        The class labels seen in ``fit``, sorted.
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + 0.2 * rng.random(1000) > 0.6).astype(np.int32)
    >>> model = mojolearn.CalibratedClassifierCV(mojolearn.GaussianNB(), cv=3).fit(X, y)
    >>> model.predict_proba(X[:5]).shape
    (5, 2)
    """
    _estimator_type = "classifier"

    def __init__(self, estimator=None, *, method="sigmoid", cv=None, n_jobs=None, ensemble=True):
        # Stored as given; `_validate_params` checks them at fit (scikit-learn's convention).
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble

    def fit(self, X, y):
        """Fit the classifier and its calibrators over the cross-validation folds.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array of shape (n_samples,)
            Class labels; fewer than two classes raise ``ValueError``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            When ``cv`` asks for more folds than some class has members.
        """
        return _TREES(self, "trees_meta_fit", (X, y, None), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Calibrated class probabilities.

        With ``ensemble=True`` the average of every fold's calibrated
        probabilities.

        Returns
        -------
        array of shape (n_samples, n_classes)
        """
        return _TREES(self, "trees_meta_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest calibrated probability.

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_meta_predict", (X,), _TREES_CONTEXT)


# The meta-estimators' native owner (bindings/trees_public_meta.mojo): each
# class's schema.
_native_trees_bridge.register_meta(VotingClassifier, "voting_classifier")
_native_trees_bridge.register_meta(VotingRegressor, "voting_regressor")
_native_trees_bridge.register_meta(StackingClassifier, "stacking_classifier")
_native_trees_bridge.register_meta(StackingRegressor, "stacking_regressor")
_native_trees_bridge.register_meta(MultiOutputRegressor, "multioutput_regressor")
_native_trees_bridge.register_meta(MultiOutputClassifier, "multioutput_classifier")
_native_trees_bridge.register_meta(OneVsRestClassifier, "ovr")
_native_trees_bridge.register_meta(CalibratedClassifierCV, "calibrated")


# -------------------------------------------------------------------- SHAP
# TreeExplainer: the `shap` package's exact path-dependent TreeSHAP
# (`shap/explainers/_tree.py` -> `shap/cext/tree_shap.h`), restated in
# xtrees/shap.mojo (float32, per-leaf paths, fixed folds) over this
# library's flat forests (RandomForest*, ExtraTrees*, DecisionTree*) and
# DART, on the device on GPU installs. DEVIATION: the node cover is the
# count of BACKGROUND rows (`data`, required) reaching each node, since the
# flat forest stores no instance counts. KernelExplainer and
# PermutationExplainer: cuML's `explainer/kernel_shap.cu` and
# `permutation_shap.cu` build the coalition datasets; the sampling and the
# solve follow `shap`'s KernelExplainer (`_kernel.py`: full enumeration of
# the small coalition sizes, then weighted sampling of the rest, the
# efficiency-constrained weighted least squares) and PermutationExplainer
# (`_permutation.py`: forward then backward passes over each permutation).
# Lane cgr2-metrics-shap: every step but the model call runs on the device
# over a chunk of rows at once (xtrees/agnostic*.mojo). DEVIATIONS: draws
# come from the lane's counter RNG; KernelExplainer's sampled coalitions are
# pairs (a draw and its complement) without de-duplication (cuML's
# schedule), carries no l1 feature selection (`l1_reg` is refused) and
# treats every feature as varying.
def _trees_forest_arrays(est):
    """(offsets, colid, quesval, left, leaves, k, scale) of a fitted flat
    forest, or None."""
    if not hasattr(est, "_offsets"):
        return None
    k = int(getattr(est, "_num_outputs", 1))
    return (est._offsets, est._colid, est._quesval, est._left_child, est._leaves, k,
            1.0 / int(est._n_trees))


def _trees_dart_forest(b, model, K):
    """(arrays, tscale) of a DART model as ONE flat forest, in boosting order:
    the node arrays joined as bytes (memory copies, no per-node Python), a
    multiclass tree's leaf values spread into column j % K natively
    (x_trees_spread_leaves). The words the per-node Python lists built (lane
    apple-fast-py2mojo-trees); O(trees) Python for the offsets and scales."""
    offs = [0]
    for tree in model.trees_:  # glue: tree offsets over the fitted trees
        to = tree._offsets
        if len(to) != 2:
            raise ValueError("a DART tree must hold one tree")
        offs.append(offs[-1] + int(to[1]) - int(to[0]))

    def join(arrs, conv, dtype):
        raw = b"".join(conv(a, ndim=1, name="tree")[0].tobytes() for a in arrs)  # glue: joins the fitted trees byte blocks
        return frombytes(raw, dtype, (len(raw) // 4,))

    col = join([t._colid for t in model.trees_], as_i32_c, "<i4")  # glue: tree offsets over the fitted trees
    q = join([t._quesval for t in model.trees_], as_f32_c, "<f4")  # glue: tree offsets over the fitted trees
    lc = join([t._left_child for t in model.trees_], as_i32_c, "<i4")  # glue: tree offsets over the fitted trees
    vals = join(model.tree_values_, as_f32_c, "<f4")
    if K > 1:
        nn = len(vals)
        lv = empty((nn * K,), "<f4")
        offs_a = Array.from_list(offs, "<i4")
        b.x_trees_spread_leaves(addr_ro(vals, name="values"), addr_ro(offs_a, name="offsets"), addr(lv, name="leaves"),
                                [nn, len(model.trees_), K])
    else:
        lv = vals
    tscale = Array.from_list([float(c) for c in model.tree_coefs_], "<f4")  # glue: tree coefficients as float32 words
    return (Array.from_list(offs, "<i4"), col, q, lc, lv), tscale


# GPU installs run it on the device (xtrees/shap_device.mojo), CPU-only
# installs on the host (xtrees/shap_host.mojo); both run the units of
# xtrees/shap.mojo in the same fixed orders, so the values are the same
# bits on every column. Float32 throughout (Metal has no float64).
class TreeExplainer(_TreesEnsembleBase):
    """Exact SHAP values for mojolearn tree models.

    Computes the exact path-dependent TreeSHAP of the ``shap`` package for a
    fitted ``RandomForestClassifier``, ``RandomForestRegressor``,
    ``ExtraTreesClassifier``, ``ExtraTreesRegressor``,
    ``DecisionTreeClassifier``, ``DecisionTreeRegressor``, ``DARTClassifier`` or
    ``DARTRegressor``. On GPU installs it runs on the device. Values are
    float32.

    Differs from ``shap``: the tree models store no per-node training counts,
    so ``data`` is required and each node's cover is the number of background
    rows that reach it.

    Parameters
    ----------
    model : estimator
        A fitted mojolearn tree model from the list above. Any other model
        raises ``NotImplementedError``; an unfitted DART model raises
        ``RuntimeError``.
    data : array of shape (n_background, n_features)
        Background rows that define the node covers. Required: None raises
        ``NotImplementedError``.
    feature_perturbation : {"tree_path_dependent", "auto"}, default "tree_path_dependent"
        Only the path-dependent algorithm is available; ``"interventional"``
        raises ``NotImplementedError``.
    model_output : {"raw"}, default "raw"
        Only the raw model output is explained; other values raise
        ``NotImplementedError``.

    Attributes
    ----------
    expected_value : float or array of shape (n_outputs,)
        The base value the SHAP values add up from: a float for one output,
        else a float32 array.
    n_outputs_ : int
        Number of model outputs explained.
    n_features_in_ : int
        Number of features in ``data``.

    Raises
    ------
    NotImplementedError
        When a root-to-leaf path splits on more than 255 distinct features.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.RandomForestRegressor().fit(X, y)
    >>> explainer = mojolearn.TreeExplainer(model, X[:200])
    >>> explainer.shap_values(X[:10]).shape
    (10, 8)
    """

    def __init__(self, model, data=None, *, feature_perturbation="tree_path_dependent", model_output="raw"):
        if feature_perturbation not in ("tree_path_dependent", "auto"):
            _refuse(f"feature_perturbation={feature_perturbation!r}", "the interventional algorithm is not"
                    " carried; the path-dependent one is.")
        if model_output != "raw":
            _refuse(f"model_output={model_output!r}", "only the raw model output is explained.")
        if data is None:
            _refuse("data=None", "the flat forests store no node sample counts, so the cover comes from a"
                    " background dataset; pass data=.")
        self.model = model
        self.data = data
        self.feature_perturbation = feature_perturbation
        self.model_output = model_output
        self.numeric_mode = getattr(model, "numeric_mode", None)
        bg, _ = as_f32_c(data, ndim=2, name="data")
        self._bg = bg
        init = 0.0
        if isinstance(model, _DARTBase):
            if not hasattr(model, "trees_"):
                raise RuntimeError("the model is not fitted yet")
            K = int(getattr(model, "n_classes_", 1))
            init = float(model.init_score_) if K == 1 else [float(v) for v in model.init_score_]  # glue: class start scores as floats
            # one forest of every boosted tree, in boosting order, each tree
            # scaled by its coefficient; a multiclass tree scores one class:
            # its leaf values in column j % K.
            arrays, tscale = _trees_dart_forest(self._bind(), model, K)
            k = K
        else:
            fa = _trees_forest_arrays(model)
            if fa is None:
                _refuse(f"TreeExplainer over {type(model).__name__}", "the explainer reads the flat forests"
                        " (RandomForest*, ExtraTrees*, DecisionTree*) and DART models.")
            arrays = fa[:5]
            k = fa[5]
            tscale = full((len(arrays[0]) - 1,), fa[6], "<f4")
        self.n_outputs_ = k
        n_trees = len(arrays[0]) - 1
        n_nodes = int(arrays[0].tolist()[-1])
        if n_trees < 1 or n_nodes < 1:
            raise ValueError("TreeExplainer: the model holds no trees")
        ev = Array.from_list(init, "<f4") if isinstance(init, list) else full((k,), init, "<f4")
        cover = zeros((n_nodes,), "<i4")
        meta = zeros((3,), "<i4")
        native = self._bind()
        prep_args = ([addr_ro(a, name="forest") for a in arrays],  # glue: forest array addresses for the binding
                     addr_ro(tscale, name="tscale"), addr_ro(bg, name="data"),
                     addr(cover, name="cover"), addr(ev, name="ev"), addr(meta, name="meta"),
                     [bg.shape[0], bg.shape[1], n_trees, k, n_nodes])
        # lane gap-shap-nb: the device binding's retained prepare uploads the
        # forest once, keeps the device cover and builds the leaf table here;
        # `shap_values` then uploads X only (xtrees/shap_device.mojo). The
        # same outputs and the same bits as the plain prepare + shap call.
        retain = getattr(native, "x_trees_tree_shap_prepare_retain", None)
        self._shap_cache = None
        self._shap_closed = False
        if callable(retain):
            self._shap_cache = retain(*prep_args)
            self._shap_cache_finalizer = weakref.finalize(
                self, native.x_trees_tree_shap_cache_release, self._shap_cache)
        else:
            native.x_trees_tree_shap_prepare(*prep_args)
        slots, depth, _ = meta.tolist()
        need = min(depth, slots) + 1
        width = 8
        while width < need:
            width *= 2
        if width > 256:
            _refuse("a leaf path of more than 255 distinct split features", "the device path is compiled up to"
                    " 255 features per root-to-leaf path.")
        self._forest = arrays
        self._tscale = tscale
        self._cover = cover
        self._shape = (n_trees, k, n_nodes, int(slots), width)
        self.expected_value = ev.tolist()[0] if k == 1 else ev
        self.n_features_in_ = bg.shape[1]
        # T44/C51 (the host binding's opt-in snapshot; the device binding's
        # retained prepare above replaces it there) owns an immutable native
        # snapshot for this explainer. A model refit cannot leave pointers
        # into mutable/released model arrays.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        enabled = getattr(native, "x_trees_tree_shap_cache_enabled", None)
        if self._shap_cache is None and callable(enabled) and enabled():
            self._shap_cache = native.x_trees_tree_shap_cache_create(
                [addr_ro(a, name="forest") for a in self._forest],  # glue: five model addresses
                addr_ro(self._tscale, name="tscale"), addr_ro(self._cover, name="cover"),
                [self.n_features_in_, n_trees, k, n_nodes])
            self._shap_cache_finalizer = weakref.finalize(
                self, native.x_trees_tree_shap_cache_release, self._shap_cache)

    def close(self):
        """Release the explainer's native resources.

        After ``close``, ``shap_values`` raises ``RuntimeError``.
        """
        finalizer = getattr(self, "_shap_cache_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()
        self._shap_cache = None
        self._shap_closed = True

    def shap_values(self, X):
        """SHAP values of the rows of ``X``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Rows to explain, converted to float32. A feature count different from
            ``data`` raises ``ValueError``.

        Returns
        -------
        array, float32
            Shape (n_samples, n_features) for one output, else
            (n_samples, n_features, n_outputs).
        """
        if getattr(self, "_shap_closed", False):
            raise RuntimeError("this TreeExplainer has been closed")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        if d != self.n_features_in_:
            raise ValueError(f"X has {d} features, data has {self.n_features_in_}")
        n_trees, k, n_nodes, slots, width = self._shape
        phi = zeros((n * d * k,), "<f4")
        if self._shap_cache is not None:
            self._bind().x_trees_tree_shap_cache_values(
                self._shap_cache, addr_ro(Xa, name="X"), addr(phi, name="phi"),
                [n, slots, width])
            return phi.reshape((n, d)) if k == 1 else phi.reshape((n, d, k))
        self._bind().x_trees_tree_shap([addr_ro(a, name="forest") for a in self._forest],  # glue: forest array addresses for the binding
                                       addr_ro(self._tscale, name="tscale"), addr_ro(self._cover, name="cover"),
                                       addr_ro(Xa, name="X"), addr(phi, name="phi"),
                                       [n, d, n_trees, k, n_nodes, slots, width])
        return phi.reshape((n, d)) if k == 1 else phi.reshape((n, d, k))


#: xtrees/agnostic.mojo AGN_BUDGET: synthetic words per explainer chunk
_AGN_BUDGET = 1 << 25
#: MOJOLEARN_IDN_PSHAP_RESIDENT (lane/neighbors-small-ports, default off;
#: `x_trees_pshap_linear_ok`): PermutationExplainer over this library's
#: linear regressors keeps a chunk's synthetic rows on the device
#: (xtrees/agnostic_device.mojo `pshap_values_linear`), so a chunk is bounded
#: by device memory, not by a host matrix: 2^28 float32 words (1 GiB of
#: synthetic rows), a memory bound that fits every supported GPU, no shape rule.
_AGN_BUDGET_RESIDENT = 1 << 28
#: MOJOLEARN_IDN_SHAP_DEVICE_MODEL (lane fam2-forests; an IDENTICAL GPU
#: build's switch, `x_trees_fast_switches` bit 64, -D
#: MOJOLEARN_IDN_SHAP_DEVICE_MODEL_OFF clears it): Kernel/Permutation SHAP
#: over one of this library's flat forests evaluate the model on the device
#: inside the explainer (xtrees/agnostic_device.mojo `model_kernel`): no
#: synthetic matrix, no model call per chunk. Moves no bit.
_AGN_IDN_DEVICE_MODEL = 64
#: MOJOLEARN_KSHAP_FAST_DEVICE_MODEL / MOJOLEARN_PSHAP_FAST_DEVICE_MODEL
#: (lane apple-fast-round2, FAST + Apple experiments, default off; bits 1024
#: and 2048 of `x_trees_fast_switches`): the FAST KernelExplainer /
#: PermutationExplainer take the device-model route above instead of
#: downloading every chunk's synthetic matrix to call the model's predict
#: (xtrees/agnostic_device.mojo has the cost reasoning). The model's
#: predict on the host inside a GPU explain is host-route debt; this route
#: removes it for this library's flat forests.
_KSHAP_FAST_DEVICE_MODEL = 1024
_PSHAP_FAST_DEVICE_MODEL = 2048


def _agn_device_forest(explainer, model):
    """(arrays, k, n_trees, rf_input) when `model` is one of this library's
    flat forests whose predict is the strict increasing-tree device kernel
    (IDENTICAL, the `sequential` engine served resident) and the explainer's
    binding carries MOJOLEARN_IDN_SHAP_DEVICE_MODEL; else None (the model
    callback). The arrays are the snapshot the forest's own predict
    validated and froze (`_prepare_resident_forest`). rf_input: the random
    forest's predict flushes the compared feature, ExtraTrees' does not."""
    if _trees_fast_tier(explainer):
        # FAST experiments: one bit per explainer (routing only, no compute)
        bit = _PSHAP_FAST_DEVICE_MODEL if isinstance(explainer, PermutationExplainer) else _KSHAP_FAST_DEVICE_MODEL
        if not _trees_switch(explainer, bit):
            return None
        fast = True
    elif not _trees_build_switch(explainer, _AGN_IDN_DEVICE_MODEL):
        return None
    else:
        fast = False
    m = getattr(model, "_random_inner", None)
    if m is None:
        m = model
    if isinstance(m, (RandomForestClassifier, RandomForestRegressor)):
        rf_input = 1
    elif isinstance(m, (_ET_CLS, ExtraTreesRegressor)):
        rf_input = 0
    else:
        return None
    if not type(m).__module__.startswith(__package__ + "."):   # a subclass outside the library may predict otherwise
        return None
    if not hasattr(m, "_offsets") or not callable(getattr(m, "_ordered_resident_auto", None)):
        return None
    try:
        if fast:
            # FAST: the forest's own predict may be parallel_groves; the
            # device walk is the strict increasing-tree sum over the same
            # validated snapshot (no bits requirement in FAST).
            # _prepare_resident_forest refuses (ValueError) a forest with
            # neither resident route, which keeps the callback.
            if m._effective_mode() not in ("fast", "identical"):
                return None
        elif m._effective_mode() != "identical" or not m._ordered_resident_auto():
            return None
        m._prepare_resident_forest()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return None
    names = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    arrays = tuple(getattr(m, name) for name in names)  # glue: the five model arrays by name
    return arrays, int(m._num_outputs), int(m._n_trees), rf_input


def _f64_word(x):
    """The binary64 bits of x as a signed int (a binding param word)."""
    import struct
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def _trees_model_fn(model):
    """The function an agnostic explainer explains: a callable as given, a
    classifier's predict_proba, else predict."""
    if callable(model) and not hasattr(model, "predict"):
        return model
    if getattr(model, "_estimator_type", None) == "classifier" and hasattr(model, "predict_proba"):
        return model.predict_proba
    return model.predict


class _AgnosticExplainer(_TreesEnsembleBase):
    def __init__(self, model, data, random_state):
        self.model = model
        self.data = data
        self.random_state = random_state
        self.numeric_mode = getattr(model, "numeric_mode", None)
        _trees_seed(random_state)
        self._f = _trees_model_fn(model)
        bg, _ = as_f32_c(data, ndim=2, name="data")
        self._bg = bg
        out = self._eval(bg)
        self.n_outputs_ = out.shape[1]
        ev = zeros((self.n_outputs_,), "<f8")
        self._bind().x_trees_block_mean(addr_ro(out, name="y"), addr(ev, name="ev"),
                                        [1, bg.shape[0], self.n_outputs_])
        self._fnull = ev
        self.expected_value = ev.tolist()[0] if self.n_outputs_ == 1 else ev
        self.n_features_in_ = bg.shape[1]
        dev = _agn_device_forest(self, model)
        if dev is not None and (dev[1] != self.n_outputs_ or int(getattr(model, "n_features_in_", -1)) != bg.shape[1]):
            dev = None
        self._dev = dev

    def _model_load(self, b):
        """MOJOLEARN_IDN_SHAP_DEVICE_MODEL: uploads the explained forest and
        the background for this `shap_values` call (`x_trees_agn_model_load`);
        True when the chunks run the model on the device, False for the
        callback route. The caller releases it (`x_trees_agn_model_release`)."""
        dev = getattr(self, "_dev", None)
        if dev is None:
            return False
        arrays, k, n_trees, rf_input = dev
        faddr = [addr_ro(a, name="forest") for a in arrays]  # glue: one address per model array (five)
        b.x_trees_agn_model_load(faddr, addr_ro(self._bg, name="data"),
                                 [n_trees, arrays[1].size, k, self._bg.shape[1], self._bg.shape[0], rf_input])
        return True

    def _eval(self, X):
        """The model output on X as a float32 (n, k) Array."""
        out = self._f(X)
        n = X.shape[0]
        return _trees_output_2d(out, n)

    def _chunk(self, per_row, n):
        """Rows per chunk: as many as keep a chunk's synthetic matrix under
        the budget (xtrees/agnostic.mojo AGN_BUDGET words), at least one."""
        return max(1, min(n, _AGN_BUDGET // max(1, per_row)))

    def _model_rows(self, syn, rows, d):
        """The model on a chunk's synthetic matrix: float32 (rows, k)."""
        return self._eval(syn.reshape((rows, d)))

    def _check(self, X):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, data has {self.n_features_in_}")
        return Xa

    def _shape(self, phi, n, d):
        k = self.n_outputs_
        return phi.reshape((n, d)) if k == 1 else phi.reshape((n, d, k))


class KernelExplainer(_AgnosticExplainer):
    """Model-agnostic Kernel SHAP values.

    Follows the ``shap`` package's ``KernelExplainer``: every coalition size
    small enough is enumerated in full, the rest are sampled with Kernel SHAP
    weights, and the values come from the efficiency-constrained weighted
    least squares. Every step except the model call runs on the device on GPU
    installs. The model is called on synthetic rows built from ``data``, so
    explaining ``n`` rows makes ``n * n_coalitions * n_background`` model
    evaluations.

    Differs from ``shap``: random draws come from mojolearn's own seeded
    generator, so values differ from ``shap``'s for the same seed; sampled
    coalitions come in complementary pairs and are not de-duplicated (as in
    cuML); there is no l1 feature selection; every feature is treated as
    varying.

    Parameters
    ----------
    model : estimator or callable
        A fitted mojolearn estimator, or a function mapping an
        (n, n_features) array to outputs. For a classifier with
        ``predict_proba`` the probabilities are explained, otherwise
        ``predict``.
    data : array of shape (n_background, n_features)
        Background rows that stand in for missing features.
    link : {"identity", "logit"}, default "identity"
        Applied to the model output before it is explained, by name; a link
        object or any other name raises ``NotImplementedError``.
    random_state : int or None, default None
        Seed for the coalition sampling: None (seed 0) or a nonnegative int.
        Anything else, including a ``RandomState`` object, raises
        ``ValueError``.

    Attributes
    ----------
    expected_value : float or array of shape (n_outputs,)
        The mean model output over ``data``, through ``link``: a float for one
        output, else a float64 array.
    n_outputs_ : int
        Number of model outputs explained.
    n_features_in_ : int
        Number of features in ``data``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.RandomForestRegressor().fit(X, y)
    >>> explainer = mojolearn.KernelExplainer(model, X[:50], random_state=0)
    >>> explainer.shap_values(X[:10]).shape
    (10, 8)
    """

    def __init__(self, model, data, *, link="identity", random_state=None):
        if link not in ("identity", "logit"):
            _refuse(f"link={link!r}", "the identity and logit links are carried (by name; a link object is not).")
        self.link = link
        super().__init__(model, data, random_state)
        if link == "logit":
            # shap KernelExplainer: expected_value = link(fnull); the link meets
            # each output AFTER the background mean (fx, fnull, ey).
            self._fnull = self._link(self._fnull)
            self.expected_value = self._fnull.tolist()[0] if self.n_outputs_ == 1 else self._fnull

    def _link(self, a):
        """`a` (float64 Array) through the link, in place; returned."""
        if self.link == "logit":
            self._bind().x_trees_logit(addr(a, name="linked"), [len(a)])
        return a

    def _schedule(self, M, nsamples):
        """shap `KernelExplainer.explain`'s coalition schedule, built in Mojo
        (`x_trees_kshap_schedule`, xtrees/api_pyapi.mojo; lane cgr4-py-compute):
        (m, nfixed, nfull, npaired, tables, L, wrand_bits) with tables =
        (size_off Int64, size_w float64, cdf float64). Every mask is made on
        the device from these (xtrees/agnostic.mojo kshap_mask_unit)."""
        h = max(1, M // 2)
        tables = (zeros((h + 1,), "<i8"), zeros((h,), "<f8"), zeros((h,), "<f8"))
        out = zeros((6,), "<i8")
        if M > 1:
            taddrs = tuple(addr(t, name="schedule") for t in tables)  # glue: the three schedule table addresses
            self._bind().x_trees_kshap_schedule(taddrs, addr(out, name="schedule"), [M, nsamples])  # cpu-route: coalition-size weight schedule over M/2 sizes, launch parameters only
        m, nfixed, nfull, npaired, L, wbits = out.tolist()
        return m, nfixed, nfull, npaired, tables, L, wbits

    def shap_values(self, X, nsamples="auto", l1_reg="auto"):
        """Kernel SHAP values of the rows of ``X``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Rows to explain, converted to float32. A feature count different from
            ``data`` raises ``ValueError``.
        nsamples : int or "auto", default "auto"
            Number of coalitions per row. ``"auto"`` is ``2 * n_features + 2048``.
            With 30 features or fewer it is capped at ``2**n_features - 2``.
        l1_reg : {"auto", False, 0}, default "auto"
            Accepted for compatibility; no l1 feature selection is done, and any
            other value raises ``NotImplementedError``.

        Returns
        -------
        array, float64
            Shape (n_samples, n_features) for one output, else
            (n_samples, n_features, n_outputs).
        """
        if l1_reg not in ("auto", False, 0):
            _refuse(f"l1_reg={l1_reg!r}", "no l1 feature selection is carried.")
        Xa = self._check(X)
        n, d = Xa.shape
        M = d
        ns = 2 * M + 2048 if nsamples == "auto" else int(nsamples)
        if M <= 30:
            ns = min(ns, 2 ** M - 2)
        seed = _trees_seed(self.random_state)
        k = self.n_outputs_
        b = self._bind()
        nb = self._bg.shape[0]
        m, nfixed, nfull, npaired, tables, L, wbits = self._schedule(M, max(ns, 0))
        taddr = tuple(addr_ro(t, name="schedule") for t in tables)  # glue: forest array addresses for the binding
        phi = zeros((max(n * d * k, 1),), "<f8")
        if n == 0:
            return self._shape(zeros((0,), "<f8"), 0, d)
        fx = self._eval(Xa)                       # the explained rows, one model call
        x0, f0, p0 = addr_ro(Xa, name="X"), addr_ro(fx, name="fx"), addr(phi, name="phi")
        fnull = self._fnull
        R = self._chunk(m * nb * d, n)
        # MOJOLEARN_KSHAP_FAST_DEVICE_MODEL: a loaded device model (self._dev,
        # only set on FAST when that bit is on) wins over the batched download
        if m > 0 and _trees_switch(self, _KSHAP_FAST_BATCH) and getattr(self, "_dev", None) is None:
            self._batched(b, Xa, fx, taddr, phi, n, d, k, nb, m, nfixed, nfull, npaired, L, seed, wbits, R)
            return self._shape(phi, n, d)
        if self._model_load(b):
            # the forest on the device: masks -> model -> means -> solve per
            # chunk, nothing crossing but the chunk's rows and phi
            link = 1 if self.link == "logit" else 0
            nl = addr_ro(fnull, name="fnull")
            try:
                for r0 in range(0, n, R):  # glue: chunk loop (one binding call per chunk)
                    rows = min(R, n - r0)
                    b.x_trees_kshap_solve_model(x0 + 4 * r0 * d, f0 + 4 * r0 * k, nl, taddr, p0 + 8 * r0 * d * k,
                                                [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits, k, link])
            finally:
                b.x_trees_agn_model_release()
            return self._shape(phi, n, d)
        reuse = _trees_build_switch(self, _AGN_IDN_SYN_POOL)   # one synthetic buffer for every chunk
        syn = None
        try:
            for r0 in range(0, n, R):  # glue: one binding call per row chunk
                rows = min(R, n - r0)
                params = [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits]
                out = None
                if m > 0:
                    if not reuse or syn is None or syn.size != rows * m * nb * d:
                        syn = None
                        syn = empty((rows * m * nb * d,), "<f4")
                    b.x_trees_kshap_synth(x0 + 4 * r0 * d, addr_ro(self._bg, name="data"), taddr,
                                          addr(syn, name="synthetic"), params)
                    out = self._model_rows(syn, rows * m * nb, d)
                    if not reuse:
                        syn = None
                b.x_trees_kshap_solve(addr_ro(out, name="y") if out is not None else 0, f0 + 4 * r0 * k,
                                      addr_ro(fnull, name="fnull"), taddr, p0 + 8 * r0 * d * k,
                                      params + [k, 1 if self.link == "logit" else 0])
        finally:
            if reuse:
                _agn_pool_release(b)   # the pooled device buffer does not outlive the explanation
        return self._shape(phi, n, d)


    def _batched(self, b, Xa, fx, taddr, phi, n, d, k, nb, m, nfixed, nfull, npaired, L, seed, wbits, R):
        """MOJOLEARN_KSHAP_FAST_BATCH: one host buffer for every chunk's
        synthetic rows, each chunk's linked background means into one
        float64 (n, m, k) block, then the solve over as many rows per call
        as the budget holds (the same words as the per-chunk solve)."""
        link = 1 if self.link == "logit" else 0
        ey = zeros((n * m * k,), "<f8")
        e0 = addr(ey, name="ey")
        x0, bg0 = addr_ro(Xa, name="X"), addr_ro(self._bg, name="data")
        # TOMBSTONE: MOJOLEARN_KSHAP_FAST_OVERLAP (DROPPED-noise) deleted 2026-10-09 on lane/owed-deletions-D1 (`_overlapped`);
        # code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_KSHAP_FAST_OVERLAP.patch.
        syn = empty((R * m * nb * d,), "<f4")
        for r0 in range(0, n, R):  # glue: one binding call per row chunk
            rows = min(R, n - r0)
            if rows < R:
                syn = empty((rows * m * nb * d,), "<f4")
            b.x_trees_kshap_synth(x0 + 4 * r0 * d, bg0, taddr, addr(syn, name="synthetic"),
                                  [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits])
            out = self._model_rows(syn, rows * m * nb, d)
            b.x_trees_kshap_means(addr_ro(out, name="y"), e0 + 8 * r0 * m * k, [rows, nb, m, k, link])
            del out
        del syn
        f0, p0 = addr_ro(fx, name="fx"), addr(phi, name="phi")
        nl = addr_ro(self._fnull, name="fnull")
        S = self._chunk(m * d, n)
        for s0 in range(0, n, S):  # glue: solve-batch loop (one binding call per batch)
            rows = min(S, n - s0)
            b.x_trees_kshap_solve_ey(e0 + 8 * s0 * m * k, f0 + 4 * s0 * k, nl, taddr, p0 + 8 * s0 * d * k,
                                     [rows, nb, d, nfixed, m, nfull, L, npaired, s0, seed, wbits, k, link])


class PermutationExplainer(_AgnosticExplainer):
    """Model-agnostic Permutation SHAP values.

    Follows the ``shap`` package's ``PermutationExplainer``: each random
    permutation of the features adds them one by one (forward), then removes
    them (backward), and each feature's marginal contribution is averaged over
    the background rows and the permutations. Every step except the model call
    runs on the device on GPU installs.

    Differs from ``shap``: random permutations come from mojolearn's own
    seeded generator, so values differ from ``shap``'s for the same seed.

    Parameters
    ----------
    model : estimator or callable
        A fitted mojolearn estimator, or a function mapping an
        (n, n_features) array to outputs. For a classifier with
        ``predict_proba`` the probabilities are explained, otherwise
        ``predict``.
    data : array of shape (n_background, n_features)
        Background rows that stand in for missing features.
    random_state : int or None, default None
        Seed for the permutations: None (seed 0) or a nonnegative int.
        Anything else, including a ``RandomState`` object, raises
        ``ValueError``.

    Attributes
    ----------
    expected_value : float or array of shape (n_outputs,)
        The mean model output over ``data``: a float for one output, else a
        float64 array.
    n_outputs_ : int
        Number of model outputs explained.
    n_features_in_ : int
        Number of features in ``data``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32)
    >>> model = mojolearn.Ridge().fit(X, y)
    >>> explainer = mojolearn.PermutationExplainer(model, X[:50], random_state=0)
    >>> explainer.shap_values(X[:10], npermutations=5).shape
    (10, 8)
    """

    def __init__(self, model, data, *, random_state=None):
        super().__init__(model, data, random_state)

    def _linear_wb(self, b):
        """MOJOLEARN_IDN_PSHAP_RESIDENT: ([w | b] float32 block, link) when
        the binding carries the switch and the explained model's predict is
        this library's linear `link(X W^T + b)` (x_linear `decision_kernel`);
        else None (the model callback). Routing and byte copies only."""
        try:
            ok = int(b.x_trees_pshap_linear_ok()) == 1
        except (AttributeError, ImportError):
            return None
        if not ok:
            return None
        m = self.model
        pred = getattr(type(m), "predict", None)
        if getattr(pred, "__qualname__", "") != "_LinearRegressorMixin.predict":
            return None
        if int(getattr(m, "n_features_in_", -1)) != self._bg.shape[1]:
            return None
        from ._expansion_linear import _coef_block   # glue: the model's own [w | b] byte copies
        intercept = m.intercept_
        if not hasattr(intercept, "shape"):
            intercept = full((1,), float(intercept), "<f4")
        wb = _coef_block(m.coef_, intercept)
        if wb.size != self.n_outputs_ * (self._bg.shape[1] + 1):
            return None
        return wb, int(m._link)

    def shap_values(self, X, npermutations=10):
        """Permutation SHAP values of the rows of ``X``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Rows to explain, converted to float32. A feature count different from
            ``data`` raises ``ValueError``.
        npermutations : int, default 10
            Number of random feature permutations per row; less than 1 raises
            ``ValueError``.

        Returns
        -------
        array, float64
            Shape (n_samples, n_features) for one output, else
            (n_samples, n_features, n_outputs).
        """
        Xa = self._check(X)
        n, d = Xa.shape
        k = self.n_outputs_
        seed = _trees_seed(self.random_state)
        npm = int(npermutations)
        if npm < 1:
            raise ValueError("npermutations must be >= 1")
        nb = self._bg.shape[0]
        b = self._bind()
        phi = zeros((max(n * d * k, 1),), "<f8")
        if n == 0:
            return self._shape(zeros((0,), "<f8"), 0, d)
        mm = npm * (2 * d + 1)
        x0, p0 = addr_ro(Xa, name="X"), addr(phi, name="phi")
        R = self._chunk(mm * nb * d, n)
        if self._model_load(b):
            # the forest on the device: permutations -> model -> means ->
            # marginals per chunk, nothing crossing but the chunk's rows and phi
            try:
                for r0 in range(0, n, R):  # glue: chunk loop (one binding call per chunk)
                    rows = min(R, n - r0)
                    b.x_trees_pshap_values_model(x0 + 4 * r0 * d, p0 + 8 * r0 * d * k, [rows, nb, d, npm, r0, seed, k])
            finally:
                b.x_trees_agn_model_release()
            return self._shape(phi, n, d)
        lin = self._linear_wb(b)
        if lin is not None:
            # IDN_PSHAP_RESIDENT: synthetic rows, the linear predict, means
            # and marginals on the device per chunk; only x rows and phi cross
            wb, link = lin
            bg0, wb0 = addr_ro(self._bg, name="data"), addr_ro(wb, name="coef")
            Rl = max(1, min(n, _AGN_BUDGET_RESIDENT // max(1, mm * nb * d)))
            for r0 in range(0, n, Rl):  # glue: chunk loop (one binding call per chunk)
                rows = min(Rl, n - r0)
                b.x_trees_pshap_values_linear(x0 + 4 * r0 * d, bg0, wb0, p0 + 8 * r0 * d * k,
                                              [rows, nb, d, npm, r0, seed, k, link])
            return self._shape(phi, n, d)
        # one synthetic buffer for every chunk
        idn_pool = _trees_build_switch(self, _AGN_IDN_SYN_POOL)
        reuse = _trees_switch(self, _KSHAP_FAST_BATCH) or idn_pool
        if _trees_switch(self, _PSHAP_DELTA):
            self._delta(b, Xa, phi, n, d, k, nb, npm, seed, R, mm)
            return self._shape(phi, n, d)
        syn = None
        try:
            for r0 in range(0, n, R):  # glue: chunk loop (one model call per chunk)
                rows = min(R, n - r0)
                params = [rows, nb, d, npm, r0, seed]
                if not reuse or syn is None or syn.size != rows * mm * nb * d:
                    syn = None
                    syn = empty((rows * mm * nb * d,), "<f4")
                b.x_trees_pshap_synth(x0 + 4 * r0 * d, addr_ro(self._bg, name="data"), addr(syn, name="synthetic"),
                                      params)
                out = self._model_rows(syn, rows * mm * nb, d)
                if not reuse:
                    syn = None
                b.x_trees_pshap_values(addr_ro(out, name="y"), p0 + 8 * r0 * d * k, params + [k])
        finally:
            if idn_pool:
                _agn_pool_release(b)   # the pooled device buffer does not outlive the explanation
        return self._shape(phi, n, d)

    def _delta(self, b, Xa, phi, n, d, k, nb, npm, seed, R, mm):
        """MOJOLEARN_PSHAP_DELTA: per chunk, the device builds only the
        synthetic rows that differ from their previous coalition's into the
        front of one reused host buffer (their count into `tot`), the model
        runs on that prefix, and the device expands the outputs back to
        every (coalition, background row)."""
        x0, bg0, p0 = addr_ro(Xa, name="X"), addr_ro(self._bg, name="data"), addr(phi, name="phi")
        # the device keeps three Int64 words per (coalition, background row):
        # bound them as if d were at least 16 (a narrow X would else index
        # up to the whole synthetic budget in Int64 triples)
        R = min(R, self._chunk(mm * nb * max(d, 16), n))
        # TOMBSTONE: MOJOLEARN_PSHAP_FAST_OVERLAP (DROPPED-slower) deleted 2026-10-09 on lane/owed-deletions-D1
        # (`_delta_overlapped`); code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_PSHAP_FAST_OVERLAP.patch.
        tot = zeros((1,), "<i8")
        t0 = addr(tot, name="count")
        syn = empty((R * mm * nb * d,), "<f4")
        for r0 in range(0, n, R):  # glue: chunk loop (one model call per chunk)
            rows = min(R, n - r0)
            params = [rows, nb, d, npm, r0, seed]
            b.x_trees_pshap_dsynth(x0 + 4 * r0 * d, bg0, addr(syn, name="synthetic"), t0, params)
            nv = int(tot.tolist()[0])
            # the first nv rows of syn, zero-copy (syn stays alive in this frame)
            front = memory_at(addr(syn, name="synthetic"), 4 * nv * d, writable=True).cast("f")
            out = self._eval(Array.from_buffer(front).reshape((nv, d)))
            del front
            b.x_trees_pshap_dvalues(x0 + 4 * r0 * d, bg0, addr_ro(out, name="y"), p0 + 8 * r0 * d * k, t0,
                                    params + [k])
            del out
