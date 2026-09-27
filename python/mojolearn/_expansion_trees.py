# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TREES LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

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
identical contract on the GPU and on their CPU host bindings; the numeric glue
between fits (weights, votes, drops) is host arithmetic in fixed order.
"""
import numbers

from . import _portable_math as math
from . import _mojolearn_rf, _mojolearn_x_trees  # noqa: F401  the bindings this door resolves; name NO other (lane_select counts > 3 as a registry)
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, as_i32_c, empty, full, zeros
from ._labels import decode_labels, encode_labels, is_bool
from ._mode import NumericModeMixin
from ._forest_protocol import forest_estimator, _forest_fit_function
from .extratrees import ExtraTreesRegressor
from .randomforest import (
    RandomForestClassifier, RandomForestRegressor, _class_weight_rows, _refuse,
)

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]


# ----------------------------------------------------------------- helpers
def _trees_f32_list(values, n, name):
    """`values` (buffer or sequence) as a list of n float32-rounded floats,
    refusing non-finite and negative entries (a sample weight)."""
    arr, _ = as_f32_c(values, ndim=1, name=name)
    out = arr.tolist()
    if len(out) != n:
        raise ValueError(f"{name} has {len(out)} entries, X has {n} rows")
    for v in out:
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not any(v > 0 for v in out):
        raise ValueError(f"{name} must have a positive total")
    return out


def _trees_colmajor(X, est):
    """X as a column-major float32 Array through the lane's own transpose, so
    the weighted fit (which takes the column-major layout) needs no base
    binding helper on a CPU-only install."""
    from . import _backend
    Xa, _ = as_f32_c(X, ndim=2, name="X")
    n, d = Xa.shape
    out = empty((d, n), "<f4")
    _backend.binding("_mojolearn_x_trees", getattr(est, "numeric_mode", None)).x_trees_transpose_f32(
        addr_ro(Xa, name="X"), addr(out, name="X^T"), [n, d])
    return Array._view_of(out, (n, d), order="F")


def _trees_weighted_rows(sample_weight, class_weight, classes, codes):
    """Per-row float32 weights: sample_weight times the class_weight row
    weight. Each product of two float32 values is exact in binary64, so the
    one rounding to float32 is the correctly rounded float32 product on every
    host."""
    n = len(codes)
    sw = _trees_f32_list(sample_weight, n, "sample_weight")
    cw = None if class_weight is None else _class_weight_rows(class_weight, classes, codes.tolist())
    if cw is not None:
        cw = cw.tolist()
        sw = [a * b for a, b in zip(sw, cw)]
    return Array.from_list(sw, "<f4")


# ----------------------------------------------------------- decision trees
# Reference: scikit-learn `sklearn/tree/_classes.py` (DecisionTreeClassifier
# :716, DecisionTreeRegressor :1100, BaseDecisionTree.fit :230). The learner
# is the RF builder (`ensemble/`, cuML's batched level algorithm) run as ONE
# tree, no bootstrap, every feature: sklearn's CART, with two differences said
# out loud: splits are searched over `n_bins` per-feature quantiles (cuML's
# rule, default 128), not every midpoint; and `splitter='random'` and
# `max_leaf_nodes` (best-first growth) are refused by name.
_DT_DEPTH = None


def _dt_common(splitter, max_features):
    if splitter != "best":
        _refuse(f"splitter={splitter!r}", "only the best-split (quantile) search"
                " exists on this tree; the random splitter is ExtraTrees'.")
    return max_features


@forest_estimator("classifier")
class DecisionTreeClassifier(RandomForestClassifier):
    """sklearn's `DecisionTreeClassifier` on the forest builder: one tree, no
    bootstrap, `max_features=None` (every feature) by default.

    `fit(X, y, sample_weight=None)` takes per-row weights (multiplied into
    `class_weight`'s row weights) through the weighted RF fit entry; that
    objective has no CPU restatement yet (ensemble/host/rf_oracle.mojo), so a
    weighted fit is GPU-only and the rf host binding refuses it by name.
    `predict_proba` is the leaf's class distribution."""

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
        _dt_common(splitter, max_features)
        super().__init__(
            n_estimators=1, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            bootstrap=False, random_state=random_state, class_weight=class_weight,
            ccp_alpha=ccp_alpha, monotonic_cst=monotonic_cst, n_bins=n_bins,
            n_streams=1, device=device, inference_engine=inference_engine,
        )
        self.splitter = splitter

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            return self._fit_with_tree_start(X, y)
        self._refresh_config()
        self._capture_fit_mode()
        self.classes_, y32 = encode_labels(y)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("y has fewer than 2 classes")
        weights = _trees_weighted_rows(sample_weight, self.class_weight, self.classes_, y32)
        binding = self._bind("_mojolearn_rf")
        weighted_fit = _forest_fit_function(binding, "rf_classifier_fit_weighted")

        def fit_fn(x_addr, y_addr, params, criterion):
            return weighted_fit(x_addr, y_addr, params, criterion, addr_ro(weights, name="weights"))
        return self._fit_arrays(_trees_colmajor(X, self), y32, self.n_classes_, fit_fn)

    def get_depth(self):
        return _trees_depth(self)

    def get_n_leaves(self):
        return _trees_n_leaves(self)


@forest_estimator("regressor")
class DecisionTreeRegressor(RandomForestRegressor):
    """sklearn's `DecisionTreeRegressor` on the forest builder: one tree, no
    bootstrap, every feature. `sample_weight` is refused by name: the
    regressor fit entry carries no row weights."""

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
        n_bins=128,
        device="gpu",
        inference_engine="auto",
    ):
        _dt_common(splitter, max_features)
        super().__init__(
            n_estimators=1, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            bootstrap=False, random_state=random_state, ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst, n_bins=n_bins, n_streams=1, device=device,
            inference_engine=inference_engine,
        )
        self.splitter = splitter

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            _refuse("DecisionTreeRegressor sample_weight", "the regressor fit entry"
                    " carries no row weights.")
        return self._fit_with_tree_start(X, y)

    def get_depth(self):
        return _trees_depth(self)

    def get_n_leaves(self):
        return _trees_n_leaves(self)


def _trees_depth(est):
    """Depth of the (first) fitted tree, from its flat nodes: children of a
    node sit at left and left + 1, a leaf has left == -1."""
    if not hasattr(est, "_offsets"):
        raise RuntimeError("this estimator is not fitted yet")
    offsets = est._offsets.tolist()
    left = est._left_child.tolist()
    lo, hi = offsets[0], offsets[1]
    depth = [0] * (hi - lo)
    best = 0
    for i in range(hi - lo):
        c = left[lo + i]
        if c != -1:
            depth[c] = depth[c + 1] = depth[i] + 1
            best = max(best, depth[i] + 1)
    return best


def _trees_n_leaves(est):
    if not hasattr(est, "_offsets"):
        raise RuntimeError("this estimator is not fitted yet")
    offsets = est._offsets.tolist()
    left = est._left_child.tolist()
    return sum(1 for i in range(offsets[0], offsets[1]) if left[i] == -1)
