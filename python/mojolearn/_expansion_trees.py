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
    "BaggingClassifier",
    "BaggingRegressor",
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
    for k, v in overrides.items():
        if k in params:
            params[k] = v
    return type(est)(**params)


def _trees_arange(n):
    return Array.from_list(list(range(n)), "<i4")


class _TreesEnsembleBase(NumericModeMixin):
    """The lane's binding (`_bind()`) and its row/column helpers."""
    _BINDING = "_mojolearn_x_trees"

    def _indices(self, n_pool, n_draw, replace, seed, stream):
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


def _trees_sub_cols(est):
    """The ensemble columns an estimator fitted on codes predicts: its
    classes_ ARE codes."""
    return Array.from_list([int(c) for c in est.classes_], "<i4")


# ------------------------------------------------------------------ Bagging
# Reference: scikit-learn `sklearn/ensemble/_bagging.py` (BaseBagging.fit
# :330, _parallel_build_estimators :120, BaggingClassifier.predict_proba
# :960, BaggingRegressor.predict :1270). Deviations: rows are always drawn by
# INDEXING (sklearn turns bootstrap counts into sample weights when the base
# estimator takes them; the regressor tree here does not); draws come from
# the lane's counter RNG (xtrees/ops.mojo), so `random_state` is an int;
# oob_score and warm_start are refused by name.
class _BaggingBase(_TreesEnsembleBase):
    def __init__(self, estimator, n_estimators, max_samples, max_features, bootstrap,
                 bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose):
        if oob_score:
            _refuse("oob_score=True", "out-of-bag scoring is not carried in pass 1.")
        if warm_start:
            _refuse("warm_start=True", "there is no incremental fit here.")
        if n_jobs is not None:
            _refuse("n_jobs", "the sub-fits run one after another on the device.")
        if verbose:
            _refuse("verbose", "nothing logs per estimator.")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
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
        _trees_seed(random_state)

    @staticmethod
    def _count(v, total, name):
        if isinstance(v, numbers.Integral) and not is_bool(v):
            k = int(v)
        elif isinstance(v, numbers.Real):
            if not 0.0 < float(v) <= 1.0:
                raise ValueError(f"{name}={v} is outside (0, 1]")
            k = int(float(v) * total)
        else:
            raise ValueError(f"{name} must be an int or a float")
        if not 1 <= k <= total:
            raise ValueError(f"{name} resolves to {k}, outside [1, {total}]")
        return k

    def _fit_bags(self, X, y_sub, sample_weight, make_default):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        seed = _trees_seed(self.random_state)
        n_rows = self._count(self.max_samples, n, "max_samples")
        n_feat = self._count(self.max_features, d, "max_features")
        base = self.estimator if self.estimator is not None else make_default()
        sw = None if sample_weight is None else as_f32_c(sample_weight, ndim=1, name="sample_weight")[0]
        self.estimators_, self.estimators_features_ = [], []
        all_rows = n_rows == n and not self.bootstrap
        all_cols = n_feat == d and not self.bootstrap_features
        for k in range(int(self.n_estimators)):
            rows = _trees_arange(n) if all_rows else self._indices(n, n_rows, self.bootstrap, seed, 2 * k)
            if all_cols:
                cols = _trees_arange(d)
            else:
                cols = self._indices(d, n_feat, self.bootstrap_features, seed, 2 * k + 1)
                if not self.bootstrap_features:
                    cols = Array.from_list(sorted(cols.tolist()), "<i4")
            Xs = Xa if (all_rows and all_cols) else self._gather(Xa, rows, cols)
            ys = y_sub(rows, all_rows)
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, k))
            if sw is None:
                est.fit(Xs, ys)
            else:
                est.fit(Xs, ys, sample_weight=sw if all_rows else self._gather_vec(sw, rows))
            self.estimators_.append(est)
            self.estimators_features_.append(cols)
        self.n_features_in_ = d
        return self

    def _check_X(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, fit saw {self.n_features_in_}")
        return Xa

    def _sub_X(self, Xa, cols, rows_all):
        if len(cols) == Xa.shape[1] and cols.tolist() == list(range(Xa.shape[1])):
            return Xa
        return self._gather(Xa, rows_all, cols)


class BaggingClassifier(_BaggingBase):
    """sklearn's `BaggingClassifier` over any mojolearn classifier (default
    `DecisionTreeClassifier`). `predict_proba` averages the members'
    `predict_proba` (a member without one votes with `predict`), float64."""
    _estimator_type = "classifier"

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator, n_estimators, max_samples, max_features, bootstrap,
                         bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)

    def fit(self, X, y, sample_weight=None):
        self.classes_, codes = encode_labels(y)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("y has fewer than 2 classes")
        return self._fit_bags(X, lambda rows, all_rows: codes if all_rows else self._gather_codes(codes, rows),
                              sample_weight, DecisionTreeClassifier)

    def predict_proba(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        acc = zeros((n * k,), "<f8")
        rows_all = _trees_arange(n)
        for est, cols in zip(self.estimators_, self.estimators_features_):
            Xs = self._sub_X(Xa, cols, rows_all)
            sub = _trees_sub_cols(est)
            if hasattr(est, "predict_proba"):
                self._acc_cols(acc, est.predict_proba(Xs), sub, n, k)
            else:
                codes = self._codes_of(est.predict(Xs), est.classes_)
                self._acc_votes(acc, codes, n, k, 1.0)
        self._scale(acc, len(self.estimators_))
        return acc.reshape((n, k))

    def predict(self, X):
        p = self.predict_proba(X)
        return decode_labels(self.classes_, self._argmax(p, p.shape[0], self.n_classes_))


class BaggingRegressor(_BaggingBase):
    """sklearn's `BaggingRegressor` over any mojolearn regressor (default
    `DecisionTreeRegressor`); `predict` is the members' mean, float64."""
    _estimator_type = "regressor"

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0,
                 bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator, n_estimators, max_samples, max_features, bootstrap,
                         bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)

    def fit(self, X, y, sample_weight=None):
        y32, _ = as_f32_c(y, ndim=1, name="y")
        return self._fit_bags(X, lambda rows, all_rows: y32 if all_rows else self._gather_vec(y32, rows),
                              sample_weight, DecisionTreeRegressor)

    def predict(self, X):
        Xa = self._check_X(X)
        n = Xa.shape[0]
        acc = zeros((n,), "<f8")
        rows_all = _trees_arange(n)
        for est, cols in zip(self.estimators_, self.estimators_features_):
            self._acc(acc, est.predict(self._sub_X(Xa, cols, rows_all)), n)
        self._scale(acc, len(self.estimators_))
        return acc
