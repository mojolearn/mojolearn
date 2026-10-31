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
import weakref

from . import _mojolearn_rf, _mojolearn_x_trees  # noqa: F401  the bindings this door resolves; name NO other (lane_select counts > 3 as a registry)
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, frombytes, full, zeros
from ._buffer import memory_at
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
import sys as _native_module_sys
from ._native_loaded_modules import connect_module as _connect_native_module
# Bagging member publication uses this owner after the module's lazy import;
# subsequent class installation extends its existing tables in place.
_connect_native_module(__name__, _native_module_sys.modules[__name__], ('_TREES_CONTEXT',))

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
        numeric_mode=None,
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
            numeric_mode=numeric_mode,
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
        return _TREES(self, "trees_public_forest_fit", (X, y, sample_weight, False, None), _TREES_CONTEXT)

    def get_depth(self):
        """Return the depth of the fitted tree."""
        return _TREES(self, "trees_public_forest_depth", (), _TREES_CONTEXT)

    def get_n_leaves(self):
        """Return the number of leaves of the fitted tree."""
        return _TREES(self, "trees_public_forest_n_leaves", (), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities: the class distribution of each row's leaf.

        Returns an array of shape (n_samples, n_classes), columns in the
        order of ``classes_``.
        """
        return _TREES(self, "trees_public_forest_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class label of each row of ``X``."""
        return _TREES(self, "trees_public_forest_predict", (X,), _TREES_CONTEXT)

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
        numeric_mode=None,
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
            numeric_mode=numeric_mode,
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
        return _TREES(self, "trees_public_forest_fit", (X, y, sample_weight, False, None), _TREES_CONTEXT)

    def get_depth(self):
        """Return the depth of the fitted tree."""
        return _TREES(self, "trees_public_forest_depth", (), _TREES_CONTEXT)

    def get_n_leaves(self):
        """Return the number of leaves of the fitted tree."""
        return _TREES(self, "trees_public_forest_n_leaves", (), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the target of each row of ``X``."""
        return _TREES(self, "trees_public_forest_predict", (X,), _TREES_CONTEXT)

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


# lane/apple-fast-trees-ensembles (2026-10-02): the Apple FAST switches of
# this module are BUILD-TIME defines of the x_trees binding (`-D
# MOJOLEARN_TE_<NAME>_OFF` turns one off; providers/python/xtrees_api_pyapi.mojo `x_trees_fast_switches` is the bit
# set the FAST + Apple build fills, 0 in every other build), read through
# the binding once per fit: no env read. IDENTICAL (the default) runs the
# code above and below them unchanged. Each is ON by default in the FAST +
# Apple build since the M3 A/B (providers/python/xtrees_api_pyapi.mojo XTREES_FAST_SWITCHES):
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
#     main took 32 for PSHAP_DELTA; providers/python/xtrees_api_pyapi.mojo XTREES_FAST_SWITCHES).
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
        return _TREES(self, "trees_bagging_members", (), _TREES_CONTEXT)

    @estimators_.setter
    def estimators_(self, value):
        return _TREES(self, "trees_bagging_members_set", (value,), _TREES_CONTEXT)

    def _slice_members(self):
        """The batched forest's trees as fitted member estimators."""
        return _TREES(self, "trees_bagging_members_slice", (), _TREES_CONTEXT)

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
    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0, numeric_mode=None):
        super().__init__(estimator, n_estimators, max_samples, max_features, bootstrap,
                         bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)
        self.numeric_mode = numeric_mode

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
        return _TREES(self, "trees_public_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Mean of the members' class probabilities, float64.

        Returns an array of shape (n_samples, n_classes), columns in the
        order of ``classes_``.
        """
        return _TREES(self, "trees_public_boost_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest mean probability."""
        return _TREES(self, "trees_public_boost_predict", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

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
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

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
    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0, numeric_mode=None):
        super().__init__(estimator, n_estimators, max_samples, max_features, bootstrap,
                         bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)
        self.numeric_mode = numeric_mode

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
        return _TREES(self, "trees_public_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Mean of the members' predictions, float64."""
        return _TREES(self, "trees_public_boost_predict", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

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
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

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
    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm="SAMME",
                 random_state=None, numeric_mode=None):
        super().__init__(estimator, n_estimators, learning_rate, random_state)
        self.algorithm = algorithm
        self.numeric_mode = numeric_mode

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
        return _TREES(self, "trees_public_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def decision_function(self, X):
        """Weighted member votes, float64.

        Returns shape (n_samples,) for two classes (positive favors
        ``classes_[1]``) and (n_samples, n_classes) otherwise.
        """
        return _TREES(self, "trees_public_boost_decision_function", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities from the decision function (softmax), float64.

        Returns an array of shape (n_samples, n_classes), columns in the
        order of ``classes_``.
        """
        return _TREES(self, "trees_public_boost_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest weighted vote."""
        return _TREES(self, "trees_public_boost_predict", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

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
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

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
    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True
    _LOSSES = {"linear": 0, "square": 1, "exponential": 2}

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss="linear",
                 random_state=None, numeric_mode=None):
        super().__init__(estimator, n_estimators, learning_rate, random_state)
        self.loss = loss
        self.numeric_mode = numeric_mode

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
        return _TREES(self, "trees_public_boost_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Weighted median of the members' predictions."""
        return _TREES(self, "trees_public_boost_predict", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

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
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

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
    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True
    _KIND = 0

    def __init__(self, *, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                 min_child_samples=20, reg_lambda=0.0, max_bin=255, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, xgboost_dart_mode=False, uniform_drop=False, drop_seed=4, random_state=None,
                 reg_alpha=0.0, max_delta_step=0.0, colsample_bytree=1.0, subsample=1.0, subsample_freq=0,
                 feature_fraction_seed=2, bagging_seed=3, numeric_mode=None):
        # The checks and the stored parameters are the native owner's
        # (bindings/trees_boost_constructor_native.mojo `dart_construct`).
        _TREES(self, "trees_dart_construct", (
            n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda, max_bin,
            drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed, random_state,
            reg_alpha, max_delta_step, colsample_bytree, subsample, subsample_freq, feature_fraction_seed,
            bagging_seed), _TREES_CONTEXT)
        # Native public, so NumericModeMixin injects no numeric_mode: the
        # constructor takes it and stores it last, as AdaBoost and Bagging do
        # (get_params/clone/set_params name it).
        self.numeric_mode = numeric_mode

    def fit(self, X, y):
        """Fit the boosted trees on ``X`` (n_samples, n_features) and target ``y``.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_public_boost_fit", (X, y, None), _TREES_CONTEXT)

    def predict(self, X):
        """Raw boosted score of each row, float64."""
        return _TREES(self, "trees_public_boost_predict", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

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
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

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
    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True
    _KIND = 1

    def __init__(self, *, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                 min_child_samples=20, reg_lambda=0.0, max_bin=255, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, xgboost_dart_mode=False, uniform_drop=False, drop_seed=4, random_state=None,
                 reg_alpha=0.0, max_delta_step=0.0, colsample_bytree=1.0, subsample=1.0, subsample_freq=0,
                 feature_fraction_seed=2, bagging_seed=3, numeric_mode=None):
        # The checks and the stored parameters are the native owner's
        # (bindings/trees_boost_constructor_native.mojo `dart_construct`).
        _TREES(self, "trees_dart_construct", (
            n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda, max_bin,
            drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed, random_state,
            reg_alpha, max_delta_step, colsample_bytree, subsample, subsample_freq, feature_fraction_seed,
            bagging_seed), _TREES_CONTEXT)
        # Native public, so NumericModeMixin injects no numeric_mode: the
        # constructor takes it and stores it last, as AdaBoost and Bagging do
        # (get_params/clone/set_params name it).
        self.numeric_mode = numeric_mode

    def fit(self, X, y):
        """Fit the boosted trees on ``X`` (n_samples, n_features) and labels ``y``.

        At least two classes are required.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_public_boost_fit", (X, y, None), _TREES_CONTEXT)

    def decision_function(self, X):
        """Raw boosted scores, float64.

        Shape (n_samples,) for two classes (the log-odds of ``classes_[1]``),
        (n_samples, n_classes) otherwise.
        """
        return _TREES(self, "trees_public_boost_decision_function", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities, float64, shape (n_samples, n_classes).

        For two classes the columns are ``[1 - p, p]`` with ``p`` the sigmoid
        of the raw score; for more, the softmax of the per-class raw scores.
        """
        return _TREES(self, "trees_public_boost_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest score."""
        return _TREES(self, "trees_public_boost_predict", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_boost_score", (X, y, sample_weight), _TREES_CONTEXT)

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
    numeric_mode : {"identical", "fast"} or None, default=None
        Numeric mode for this estimator. None uses the process default (see
        ``mojolearn.set_numeric_mode``). ``"fast"`` applies on Apple GPUs
        only.

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

    #: the native lifecycle owner (trees_public_boost_*) tracks fits and validates inputs;
    #: no Python fit-tracking or output wrappers are installed
    _mojolearn_native_public = True

    def __init__(self, n_estimators=100, *, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 sparse_output=False, n_jobs=None, random_state=None, verbose=0, warm_start=False,
                 numeric_mode=None):
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
        self.numeric_mode = numeric_mode

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
        return _TREES(self, "trees_public_boost_fit", (X, y), _TREES_CONTEXT)

    def apply(self, X):
        """Leaf reached by each row in each tree.

        Returns an int32 array of shape (n_samples, n_trees) of node ids,
        numbered within each tree.
        """
        return _TREES(self, "trees_public_boost_apply", (X,), _TREES_CONTEXT)

    def transform(self, X):
        """One-hot encode the leaves of ``X``.

        Returns a dense float64 array of shape
        (n_samples, ``n_output_features_``).
        """
        return _TREES(self, "trees_public_boost_transform", (X,), _TREES_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit on ``X``, then return ``transform(X)``."""
        return _TREES(self, "trees_public_boost_fit_transform", (X, y), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_meta_fit_transform", (X, y, fit_params), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_score", (X, y, sample_weight), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_score", (X, y, sample_weight), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    @available_if(lambda self: self.voting != "hard")
    def predict_proba(self, X):
        """Weighted average of the members' class probabilities.

        Only available with ``voting="soft"``; with ``voting="hard"`` it raises
        ``AttributeError``.

        Returns
        -------
        array of shape (n_samples, n_classes), float64
        """
        return _TREES(self, "trees_public_meta_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict class labels by weighted vote (hard) or averaged probabilities (soft).

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)

    def transform(self, X):
        """The members' outputs side by side.

        Returns
        -------
        array, float64
            With ``voting="soft"``, every member's class probabilities, shape
            (n_samples, n_estimators * n_classes); with ``voting="hard"``, every
            member's predicted class code, shape (n_samples, n_estimators).
        """
        return _TREES(self, "trees_public_meta_transform", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Weighted average of the members' predictions.

        Returns
        -------
        array of shape (n_samples,), float64
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)

    def transform(self, X):
        """Every member's prediction side by side.

        Returns
        -------
        array of shape (n_samples, n_estimators), float64
        """
        return _TREES(self, "trees_public_meta_transform", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_transform", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Predict class labels with the final estimator on the stacked outputs.

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """The final estimator's class probabilities on the stacked outputs.

        Returns
        -------
        array of shape (n_samples, n_classes), float64
        """
        return _TREES(self, "trees_public_meta_predict_proba", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Predict with the final estimator on the stacked member predictions.

        Returns
        -------
        array of shape (n_samples,)
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Every target's prediction, one column per output.

        Returns
        -------
        array of shape (n_samples, n_outputs)
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, sample_weight), _TREES_CONTEXT)

    def predict(self, X):
        """Predict every output's labels.

        Returns
        -------
        array of shape (n_samples, n_outputs)
            For numeric labels, int64 when every output's labels are integers,
            else float64.
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Every output's class probabilities.

        Returns
        -------
        list of arrays
            One float64 array of shape (n_samples, n_classes_j) per output, its
            columns in the order of ``classes_[j]``.
        """
        return _TREES(self, "trees_public_meta_predict_proba", (X,), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_meta_score", (X, y, sample_weight), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, None), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class whose binary classifier scores highest.

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities.

        Each class's positive-class probability from its clone, with every row
        normalized to sum to 1 when there are more than two classes. For two
        classes the rows are ``(1 - p, p)``.

        Returns
        -------
        array of shape (n_samples, n_classes), float64
        """
        return _TREES(self, "trees_public_meta_predict_proba", (X,), _TREES_CONTEXT)


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
        return _TREES(self, "trees_public_meta_fit", (X, y, None), _TREES_CONTEXT)

    def predict_proba(self, X):
        """Calibrated class probabilities.

        With ``ensemble=True`` the average of every fold's calibrated
        probabilities.

        Returns
        -------
        array of shape (n_samples, n_classes)
        """
        return _TREES(self, "trees_public_meta_predict_proba", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Predict the class with the highest calibrated probability.

        Returns
        -------
        array of shape (n_samples,)
            Labels from ``classes_``.
        """
        return _TREES(self, "trees_public_meta_predict", (X,), _TREES_CONTEXT)


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

    _mojolearn_native_public = True

    def __init__(self, model, data=None, *, feature_perturbation="tree_path_dependent",
                 model_output="raw", numeric_mode=None):
        return _TREES(self, "trees_explainer_init",
                      (model, data, feature_perturbation, model_output, numeric_mode), _TREES_CONTEXT)

    def close(self):
        """Release the explainer's native resources.

        After ``close``, ``shap_values`` raises ``RuntimeError``.
        """
        return _TREES(self, "trees_explainer_close", (), _TREES_CONTEXT)

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
        return _TREES(self, "trees_explainer_values", (X,), _TREES_CONTEXT)


# KernelExplainer / PermutationExplainer: every operation is one call into the
# native owner (bindings/trees_agnostic_native.mojo). It admits the model
# through its compiled model-selection plan, evaluates it there (never its
# Python predict) and drives the xtrees agnostic kernels and chunk loops.
class KernelExplainer(_TreesEnsembleBase):
    """Model-agnostic Kernel SHAP values.

    Follows the ``shap`` package's ``KernelExplainer``: every coalition size
    small enough is enumerated in full, the rest are sampled with Kernel SHAP
    weights, and the values come from the efficiency-constrained weighted
    least squares. Every step runs on the device on GPU installs. The model is called on synthetic rows built from ``data``, so
    explaining ``n`` rows makes ``n * n_coalitions * n_background`` model
    evaluations.

    Differs from ``shap``: random draws come from mojolearn's own seeded
    generator, so values differ from ``shap``'s for the same seed; sampled
    coalitions come in complementary pairs and are not de-duplicated (as in
    cuML); there is no l1 feature selection; every feature is treated as
    varying.

    Parameters
    ----------
    model : estimator
        A fitted mojolearn estimator. For a classifier with
        ``predict_proba`` the probabilities are explained, otherwise
        ``predict``. Any other model, including a Python function, raises
        ``TypeError``.
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

    _mojolearn_native_public = True

    def __init__(self, model, data, *, link="identity", random_state=None, numeric_mode=None):
        return _TREES(self, "trees_agnostic_kernel_init",
                      (model, data, random_state, link, numeric_mode), _TREES_CONTEXT)

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
        return _TREES(self, "trees_agnostic_kernel_values", (X, nsamples, l1_reg), _TREES_CONTEXT)


class PermutationExplainer(_TreesEnsembleBase):
    """Model-agnostic Permutation SHAP values.

    Follows the ``shap`` package's ``PermutationExplainer``: each random
    permutation of the features adds them one by one (forward), then removes
    them (backward), and each feature's marginal contribution is averaged over
    the background rows and the permutations. Every step runs on the device on
    GPU installs.

    Differs from ``shap``: random permutations come from mojolearn's own
    seeded generator, so values differ from ``shap``'s for the same seed.

    Parameters
    ----------
    model : estimator
        A fitted mojolearn estimator. For a classifier with
        ``predict_proba`` the probabilities are explained, otherwise
        ``predict``. Any other model, including a Python function, raises
        ``TypeError``.
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

    _mojolearn_native_public = True

    def __init__(self, model, data, *, random_state=None, numeric_mode=None):
        return _TREES(self, "trees_agnostic_permutation_init",
                      (model, data, random_state, None, numeric_mode), _TREES_CONTEXT)

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
        return _TREES(self, "trees_agnostic_permutation_values", (X, npermutations), _TREES_CONTEXT)


def _connect_explainers():
    # glue: install the native attribute descriptor once at class definition;
    # the inherited Python output getter must not surround public operations.
    if "trees_unavailable" in _TREES_CONTEXT:
        return
    from types import BuiltinFunctionType, MethodType
    from .tokenizer import _binding
    from ._native_classical_bridge import _layout
    module = _binding()._module
    attribute = module.standard_scaler_attribute_call
    factory = module.classical_attribute_descriptor
    if type(attribute) is not BuiltinFunctionType or type(factory) is not BuiltinFunctionType:
        raise TypeError("native explainers require compiled attribute entries")
    descriptor = factory(MethodType(attribute, _TREES_CONTEXT))
    layouts = _TREES_CONTEXT.setdefault("classical_public_layouts", {})
    for cls in (TreeExplainer, KernelExplainer, PermutationExplainer):  # glue: the three explainer classes, once at import
        cls.__getattribute__ = descriptor
        layouts[cls] = _layout(cls, descriptor)
    _TREES_CONTEXT["classical_public_installed"] = True
    _TREES_CONTEXT["classical_attribute_descriptor"] = descriptor


_connect_explainers()
