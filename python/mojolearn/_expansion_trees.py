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
    `class_weight`'s row weights) through the weighted RF fit entry (the
    weighted objective; its CPU restatement is ensemble/host/rf_oracle.mojo).
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


# ----------------------------------------------------------------- AdaBoost
# Reference: scikit-learn `sklearn/ensemble/_weight_boosting.py`
# (BaseWeightBoosting.fit :118, AdaBoostClassifier._boost_discrete :560 --
# SAMME, the only algorithm sklearn 1.6+ keeps --, decision_function :660,
# _compute_proba_from_decision :740; AdaBoostRegressor._boost :1030 --
# AdaBoost.R2 --, _get_median_predict :1120). The weight update, the error,
# the estimator weight and the weighted median are `xtrees/ops.mojo`
# (`samme_step`, `r2_step`, `weighted_median`); R2's weighted bootstrap is the
# lane's counter RNG, not numpy's `choice`.
def _trees_normalized_weights(sample_weight, n):
    if sample_weight is None:
        return Array.from_list([1.0 / n] * n, "<f8")
    sw = [float(v) for v in as_f32_c(sample_weight, ndim=1, name="sample_weight")[0].tolist()]
    if len(sw) != n:
        raise ValueError(f"sample_weight has {len(sw)} entries, X has {n} rows")
    if any(not math.isfinite(v) or v < 0 for v in sw):
        raise ValueError("sample_weight must be finite and nonnegative")
    total = math.fsum(sw)
    if not total > 0:
        raise ValueError("sample_weight must have a positive total")
    return Array.from_list([v / total for v in sw], "<f8")


class _AdaBoostBase(_TreesEnsembleBase):
    def __init__(self, estimator, n_estimators, learning_rate, random_state):
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if not float(learning_rate) > 0:
            raise ValueError("learning_rate must be > 0")
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        _trees_seed(random_state)

    def _w32(self, w, n):
        """The boosting weights (summing to one) times n, float32: the
        members' `sample_weight` (the criterion is scale-free; the factor
        keeps the fixed-point weight plane well inside its resolution)."""
        out = empty((n,), "<f4")
        self._bind().x_trees_scale_to_f32(addr_ro(w, name="w"), addr(out, name="w32"), [n, float(n)])
        return out

    def _check_X(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, fit saw {self.n_features_in_}")
        return Xa


class AdaBoostClassifier(_AdaBoostBase):
    """sklearn's `AdaBoostClassifier` (SAMME) over a mojolearn classifier
    that takes `sample_weight`, default `DecisionTreeClassifier(max_depth=1)`:
    each member is fitted on every row with the boosting weights (times n,
    float32) as its sample weights, as sklearn's `_boost_discrete` does."""
    _estimator_type = "classifier"

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm="SAMME",
                 random_state=None):
        if algorithm not in ("SAMME", "deprecated"):
            _refuse(f"algorithm={algorithm!r}", "SAMME.R was removed from the reference (sklearn 1.6);"
                    " SAMME is the algorithm.")
        super().__init__(estimator, n_estimators, learning_rate, random_state)
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n = Xa.shape[0]
        self.classes_, codes = encode_labels(y)
        k = self.n_classes_ = len(self.classes_)
        if k < 2:
            raise ValueError("y has fewer than 2 classes")
        seed = _trees_seed(self.random_state)
        base = self.estimator if self.estimator is not None else DecisionTreeClassifier(max_depth=1)
        w = _trees_normalized_weights(sample_weight, n)
        stats = zeros((4,), "<f8")
        self.estimators_, self.estimator_weights_, self.estimator_errors_ = [], [], []
        b = self._bind()
        m = int(self.n_estimators)
        for it in range(m):
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, it))
            est.fit(Xa, codes, sample_weight=self._w32(w, n))
            pred = as_i32_c(est.predict(Xa), ndim=1, name="predicted codes")[0]
            b.x_trees_samme_step(addr(w, name="w"), addr_ro(pred, name="pred"), addr_ro(codes, name="y"),
                                 addr(stats, name="stats"),
                                 [n, k, float(self.learning_rate), 1 if it == m - 1 else 0])
            status, alpha, err, total = stats.tolist()
            if status == 2.0:
                if not self.estimators_:
                    raise ValueError("BaseClassifier in AdaBoostClassifier ensemble is worse than random,"
                                     " ensemble can not be fit.")
                break
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)
            if status == 1.0 or not total > 0:
                break
            if it < m - 1:
                self._scale(w, total)
        self.n_features_in_ = Xa.shape[1]
        return self

    def _decision(self, Xa):
        n, k = Xa.shape[0], self.n_classes_
        acc = zeros((n * k,), "<f8")
        for est, a in zip(self.estimators_, self.estimator_weights_):
            pred = as_i32_c(est.predict(Xa), ndim=1, name="predicted codes")[0]
            self._acc_votes(acc, pred, n, k, a, -a / (k - 1))
        self._scale(acc, math.fsum(self.estimator_weights_))
        return acc

    def decision_function(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        acc = self._decision(Xa)
        if k == 2:
            v = acc.tolist()
            return Array.from_list([v[2 * i + 1] - v[2 * i] for i in range(n)], "<f8")
        return acc.reshape((n, k))

    def predict_proba(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        acc = self._decision(Xa)
        if k == 2:
            v = acc.tolist()
            d = [(v[2 * i + 1] - v[2 * i]) / 2 for i in range(n)]
            acc = Array.from_list([x for di in d for x in (-di, di)], "<f8")
        else:
            self._scale(acc, k - 1)
        self._bind().x_trees_softmax_rows(addr(acc, name="proba"), [n, k])
        return acc.reshape((n, k))

    def predict(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        acc = self._decision(Xa)
        if k == 2:
            v = acc.tolist()
            codes = Array.from_list([1 if v[2 * i + 1] - v[2 * i] > 0 else 0 for i in range(n)], "<i4")
        else:
            codes = self._argmax(acc, n, k)
        return decode_labels(self.classes_, codes)


class AdaBoostRegressor(_AdaBoostBase):
    """sklearn's `AdaBoostRegressor` (AdaBoost.R2) over any mojolearn
    regressor, default `DecisionTreeRegressor(max_depth=3)`: each member fits
    a weighted bootstrap of the rows; `predict` is the weighted median."""
    _estimator_type = "regressor"
    _LOSSES = {"linear": 0, "square": 1, "exponential": 2}

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss="linear",
                 random_state=None):
        if loss not in self._LOSSES:
            raise ValueError(f"loss must be one of {tuple(self._LOSSES)}, got {loss!r}")
        super().__init__(estimator, n_estimators, learning_rate, random_state)
        self.loss = loss

    def fit(self, X, y, sample_weight=None):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        y32, _ = as_f32_c(y, ndim=1, name="y")
        if len(y32) != n:
            raise ValueError(f"y has {len(y32)} rows, X has {n}")
        seed = _trees_seed(self.random_state)
        base = self.estimator if self.estimator is not None else DecisionTreeRegressor(max_depth=3)
        w = _trees_normalized_weights(sample_weight, n)
        stats = zeros((4,), "<f8")
        cols = _trees_arange(d)
        self.estimators_, self.estimator_weights_, self.estimator_errors_ = [], [], []
        b = self._bind()
        m = int(self.n_estimators)
        for it in range(m):
            rows = empty((n,), "<i4")
            b.x_trees_weighted_sample(addr_ro(w, name="w"), addr(rows, name="rows"), [n, n, seed, it])
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, it))
            est.fit(self._gather(Xa, rows, cols), self._gather_vec(y32, rows))
            pred, _ = as_f32_c(est.predict(Xa), ndim=1, name="prediction")
            b.x_trees_r2_step(addr(w, name="w"), addr_ro(pred, name="pred"), addr_ro(y32, name="y"),
                              addr(stats, name="stats"),
                              [n, self._LOSSES[self.loss], float(self.learning_rate), 1 if it == m - 1 else 0])
            status, alpha, err, total = stats.tolist()
            if status == 2.0:
                if not self.estimators_:
                    self.estimators_.append(est)
                    self.estimator_weights_.append(0.0)
                    self.estimator_errors_.append(err)
                break
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)
            if status == 1.0 or not total > 0:
                break
            if it < m - 1:
                self._scale(w, total)
        self.n_features_in_ = d
        return self

    def predict(self, X):
        Xa = self._check_X(X)
        n, m = Xa.shape[0], len(self.estimators_)
        preds = empty((m * n,), "<f4")
        b = self._bind()
        for j, est in enumerate(self.estimators_):
            p, _ = as_f32_c(est.predict(Xa), ndim=1, name="prediction")
            b.x_trees_put_f32(addr(preds, name="preds"), addr_ro(p, name="p"), [j * n, n])
        weights = Array.from_list([float(a) for a in self.estimator_weights_], "<f8")
        out = empty((n,), "<f4")
        b.x_trees_weighted_median(addr_ro(preds, name="preds"), addr_ro(weights, name="weights"),
                                  addr(out, name="median"), [n, m])
        return out


# --------------------------------------------------------------------- DART
# Reference: LightGBM `src/boosting/dart.hpp` (DART::DroppingTrees,
# ::Normalize, ::TrainOneIter) over `gbdt.cpp`'s boosting loop, with
# `regression_objective.hpp` (L2) and `binary_objective.hpp` (logloss,
# sigmoid 1) for the gradients. Carried: drop_rate, max_drop, skip_drop,
# uniform_drop, xgboost_dart_mode, drop_seed, the tree weights and their
# normalisation, Newton leaf values -sum(g) / (sum(h) + reg_lambda).
# DEVIATIONS: each iteration's tree is the forest builder's (level-order,
# quantile bins, `num_leaves` as cuML's `max_leaves` cap) fitted by squared
# error to -g, where LightGBM grows leaf-wise on the g/h histogram gain; the
# leaf VALUES are then LightGBM's Newton step on the rows that reach them. The
# drop draws come from the lane's counter RNG seeded by drop_seed, not
# LightGBM's LCG; the average/log-odds start is a constant outside the trees
# (never dropped), where LightGBM folds it into tree 0's bias. Multiclass is
# refused by name.
class _DARTBase(_TreesEnsembleBase):
    _KIND = 0

    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda,
                 max_bin, drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed,
                 random_state):
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if not float(learning_rate) > 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= float(drop_rate) <= 1.0 or not 0.0 <= float(skip_drop) <= 1.0:
            raise ValueError("drop_rate and skip_drop must be in [0, 1]")
        if float(reg_lambda) < 0:
            raise ValueError("reg_lambda must be >= 0")
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
        _trees_seed(random_state)
        _trees_seed(drop_seed)

    def _tree_nodes(self, tree, Xa):
        n, d = Xa.shape
        out = empty((n,), "<i4")
        self._bind().x_trees_apply(addr_ro(tree._offsets, name="offsets"), addr_ro(tree._colid, name="colid"),
                                   addr_ro(tree._quesval, name="quesval"), addr_ro(tree._left_child, name="left"),
                                   addr_ro(Xa, name="X"), addr(out, name="nodes"), [n, d, 0, 1])
        return out

    def _add(self, score, nodes, values, weight):
        self._bind().x_trees_tree_score_add(addr_ro(nodes, name="nodes"), addr_ro(values, name="values"),
                                            addr(score, name="score"), [len(nodes), float(weight)])

    def _boost(self, Xa, y32):
        n, d = Xa.shape
        b = self._bind()
        seed = _trees_seed(self.random_state)
        drop_seed = _trees_seed(self.drop_seed)
        yv = y32.tolist()
        if self._KIND == 0:
            init = math.fsum(yv) / n
        else:
            p = math.fsum(yv) / n
            if not 0.0 < p < 1.0:
                raise ValueError("y must hold both classes")
            init = float(b.x_trees_log64(p / (1.0 - p)))
        self.init_score_ = init
        score = full((n,), init, "<f8")
        g, h, target = empty((n,), "<f8"), empty((n,), "<f8"), empty((n,), "<f4")
        lr = float(self.learning_rate)
        self.trees_, self.tree_values_, self.tree_coefs_, self.tree_weights_ = [], [], [], []
        train_nodes = []
        sum_w = 0.0
        max_depth = None if self.max_depth is None or int(self.max_depth) <= 0 else int(self.max_depth)
        for it in range(int(self.n_estimators)):
            t = len(self.trees_)
            u = empty((1 + t,), "<f8")
            b.x_trees_uniform(addr(u, name="u"), [1 + t, drop_seed, it])
            uv = u.tolist()
            drop = []
            if t and not uv[0] < float(self.skip_drop):
                rate = float(self.drop_rate)
                if not self.uniform_drop:
                    inv_avg = t / sum_w if sum_w > 0 else 0.0
                    if int(self.max_drop) > 0 and sum_w > 0:
                        rate = min(rate, int(self.max_drop) * inv_avg / sum_w)
                    drop = [i for i in range(t) if uv[1 + i] < rate * self.tree_weights_[i] * inv_avg]
                else:
                    if int(self.max_drop) > 0:
                        rate = min(rate, int(self.max_drop) / t)
                    drop = [i for i in range(t) if uv[1 + i] < rate]
            for i in drop:
                self._add(score, train_nodes[i], self.tree_values_[i], -self.tree_coefs_[i])
            k = len(drop)
            if not self.xgboost_dart_mode:
                shrink = lr / (1.0 + k)
            else:
                shrink = lr if k == 0 else lr / (lr + k)
            b.x_trees_gradients(addr_ro(score, name="score"), addr_ro(y32, name="y"), addr(g, name="g"),
                                addr(h, name="h"), addr(target, name="target"), [n, self._KIND])
            tree = RandomForestRegressor(
                n_estimators=1, bootstrap=False, max_features=1.0, max_depth=max_depth,
                max_leaves=int(self.num_leaves), min_samples_leaf=int(self.min_child_samples),
                n_bins=int(self.max_bin), random_state=_trees_sub_seed(seed, it), n_streams=1,
                numeric_mode=self.numeric_mode).fit(Xa, target)
            nodes = self._tree_nodes(tree, Xa)
            n_nodes = int(tree._offsets.tolist()[1])
            values = empty((n_nodes,), "<f4")
            b.x_trees_leaf_newton(addr_ro(nodes, name="nodes"), addr_ro(g, name="g"), addr_ro(h, name="h"),
                                  addr(values, name="values"), [n, n_nodes, float(self.reg_lambda)])
            self._add(score, nodes, values, shrink)
            for i in drop:
                if not self.xgboost_dart_mode:
                    factor, wdiv = k / (k + 1.0), 1.0 / (k + 1.0)
                else:
                    factor, wdiv = k / (k + lr), 1.0 / (k + lr)
                self.tree_coefs_[i] *= factor
                self._add(score, train_nodes[i], self.tree_values_[i], self.tree_coefs_[i])
                if not self.uniform_drop:
                    sum_w -= self.tree_weights_[i] * wdiv
                    self.tree_weights_[i] *= factor
            self.trees_.append(tree)
            self.tree_values_.append(values)
            self.tree_coefs_.append(shrink)
            self.tree_weights_.append(shrink)
            sum_w += shrink
            train_nodes.append(nodes)
        self.n_features_in_ = d
        return self

    def _raw(self, X):
        if not hasattr(self, "trees_"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, fit saw {self.n_features_in_}")
        score = full((Xa.shape[0],), self.init_score_, "<f8")
        for tree, values, coef in zip(self.trees_, self.tree_values_, self.tree_coefs_):
            self._add(score, self._tree_nodes(tree, Xa), values, coef)
        return score


class DARTRegressor(_DARTBase):
    """LightGBM's `boosting='dart'` with the L2 objective. `predict` is the
    raw score, float64."""
    _estimator_type = "regressor"
    _KIND = 0

    def __init__(self, *, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                 min_child_samples=20, reg_lambda=0.0, max_bin=255, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, xgboost_dart_mode=False, uniform_drop=False, drop_seed=4, random_state=None):
        super().__init__(n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda,
                         max_bin, drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed,
                         random_state)

    def fit(self, X, y):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        y32, _ = as_f32_c(y, ndim=1, name="y")
        if len(y32) != Xa.shape[0]:
            raise ValueError(f"y has {len(y32)} rows, X has {Xa.shape[0]}")
        return self._boost(Xa, y32)

    def predict(self, X):
        return self._raw(X)


class DARTClassifier(_DARTBase):
    """LightGBM's `boosting='dart'` with the binary logloss objective.
    `predict_proba` is [1 - p, p] with p the sigmoid of the raw score (a
    two-column softmax of [0, raw]); multiclass is refused by name."""
    _estimator_type = "classifier"
    _KIND = 1

    def __init__(self, *, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                 min_child_samples=20, reg_lambda=0.0, max_bin=255, drop_rate=0.1, max_drop=50,
                 skip_drop=0.5, xgboost_dart_mode=False, uniform_drop=False, drop_seed=4, random_state=None):
        super().__init__(n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, reg_lambda,
                         max_bin, drop_rate, max_drop, skip_drop, xgboost_dart_mode, uniform_drop, drop_seed,
                         random_state)

    def fit(self, X, y):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        self.classes_, codes = encode_labels(y)
        if len(self.classes_) != 2:
            _refuse("DARTClassifier with %d classes" % len(self.classes_), "pass 1 carries the binary"
                    " objective only (multiclass needs one tree per class per iteration).")
        if len(codes) != Xa.shape[0]:
            raise ValueError(f"y has {len(codes)} rows, X has {Xa.shape[0]}")
        y32, _ = as_f32_c(codes, ndim=1, name="y")
        return self._boost(Xa, y32)

    def decision_function(self, X):
        return self._raw(X)

    def predict_proba(self, X):
        raw = self._raw(X).tolist()
        acc = Array.from_list([v for r in raw for v in (0.0, r)], "<f8")
        self._bind().x_trees_softmax_rows(addr(acc, name="proba"), [len(raw), 2])
        return acc.reshape((len(raw), 2))

    def predict(self, X):
        raw = self._raw(X).tolist()
        return decode_labels(self.classes_, Array.from_list([1 if r > 0 else 0 for r in raw], "<i4"))


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
    """sklearn's totally random trees embedding, dense output."""

    def __init__(self, n_estimators=100, *, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 sparse_output=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
        if sparse_output:
            _refuse("sparse_output=True", "the NumPy-free layer has no sparse container; the dense"
                    " one-hot is returned.")
        if n_jobs is not None or verbose or warm_start:
            _refuse("n_jobs/verbose/warm_start", "not carried in pass 1.")
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
        _trees_seed(random_state)

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            _refuse("RandomTreesEmbedding sample_weight", "the members take none.")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        seed = _trees_seed(self.random_state)
        u = empty((n,), "<f8")
        self._bind().x_trees_uniform(addr(u, name="u"), [n, seed, 0])
        yr, _ = as_f32_c(u, ndim=1, name="y")
        self.forest_ = ExtraTreesRegressor(
            n_estimators=int(self.n_estimators), max_depth=self.max_depth, max_features=1,
            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease,
            random_state=seed, numeric_mode=self.numeric_mode).fit(Xa, yr)
        f = self.forest_
        offsets, left = f._offsets.tolist(), f._left_child.tolist()
        n_trees = len(offsets) - 1
        node_col, col = [-1] * offsets[-1], 0
        for t in range(n_trees):
            for g in range(offsets[t], offsets[t + 1]):
                if left[g] == -1:
                    node_col[g] = col
                    col += 1
        self._tree_base = Array.from_list(offsets[:-1], "<i4")
        self._node_col = Array.from_list(node_col, "<i4")
        self.n_trees_ = n_trees
        self.n_output_features_ = col
        self.n_features_in_ = d
        return self

    def apply(self, X):
        """(n_samples, n_estimators) tree-relative leaf node ids, int32."""
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
        nodes = self.apply(X)
        n = nodes.shape[0]
        out = zeros((n * self.n_output_features_,), "<f8")
        self._bind().x_trees_onehot_leaves(addr_ro(nodes, name="nodes"), addr_ro(self._tree_base, name="base"),
                                           addr_ro(self._node_col, name="cols"), addr(out, name="embedding"),
                                           [n, self.n_trees_, self.n_output_features_])
        return out.reshape((n, self.n_output_features_))

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)


# ------------------------------------------------------ wrappers: helpers
def _trees_stratified_folds(codes, n_splits):
    """sklearn StratifiedKFold(n_splits, shuffle=False)._make_test_folds:
    classes in order of first appearance, per-class fold allocation from the
    sorted labels dealt round robin. Returns the test fold of each row."""
    first = {}
    for c in codes:
        if c not in first:
            first[c] = len(first)
    enc = [first[c] for c in codes]
    k = len(first)
    counts = [0] * k
    for e in enc:
        counts[e] += 1
    if n_splits > max(counts) and min(counts) < n_splits and max(counts) < n_splits:
        raise ValueError(f"n_splits={n_splits} cannot be greater than the number of members in each class")
    order = sorted(enc)
    alloc = [[0] * k for _ in range(n_splits)]
    for i in range(n_splits):
        for e in order[i::n_splits]:
            alloc[i][e] += 1
    folds = [0] * len(enc)
    for c in range(k):
        seq = [i for i in range(n_splits) for _ in range(alloc[i][c])]
        pos = 0
        for r, e in enumerate(enc):
            if e == c:
                folds[r] = seq[pos]
                pos += 1
    return folds


def _trees_kfolds(n, n_splits):
    """sklearn KFold(n_splits, shuffle=False): contiguous folds, the first
    n % n_splits one row longer."""
    folds, start = [0] * n, 0
    for i in range(n_splits):
        size = n // n_splits + (1 if i < n % n_splits else 0)
        for r in range(start, start + size):
            folds[r] = i
        start += size
    return folds


def _trees_cv(cv, default=5):
    if cv is None:
        return default
    if is_bool(cv) or not isinstance(cv, numbers.Integral) or cv < 2:
        _refuse(f"cv={cv!r}", "only an int number of folds (>= 2) is carried: the folds are sklearn's"
                " unshuffled (Stratified)KFold.")
    return int(cv)


def _trees_fold_rows(folds, i):
    tr = [r for r, f in enumerate(folds) if f != i]
    te = [r for r, f in enumerate(folds) if f == i]
    return Array.from_list(tr, "<i4"), Array.from_list(te, "<i4")


def _trees_estimators(estimators):
    if not estimators:
        raise ValueError("estimators must be a non-empty list of (name, estimator)")
    names = [n for n, _ in estimators]
    if len(set(names)) != len(names):
        raise ValueError("estimator names must be unique")
    active = [(n, e) for n, e in estimators if e != "drop"]
    if not active:
        raise ValueError("all estimators are 'drop'")
    return active


def _trees_output_2d(x, n):
    """A member's output as a float32 (n, c) Array (a 1-D output is c = 1)."""
    a, _ = as_f32_c(x, ndim=2, name="output") if len(getattr(x, "shape", ())) == 2 else \
        (as_f32_c(x, ndim=1, name="output")[0].reshape((n, 1)), None)
    return a


class _TreesWrapperBase(_TreesEnsembleBase):
    def _check_X(self, X):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, fit saw {self.n_features_in_}")
        return Xa

    def _place(self, dst, n_rows, n_cols, out, rows, col0):
        m = len(rows)
        blk = _trees_output_2d(out, m)
        self._bind().x_trees_scatter(addr(dst, name="dst"), addr_ro(blk, name="src"), addr_ro(rows, name="rows"),
                                     [n_rows, n_cols, m, blk.shape[1], col0])
        return blk.shape[1]

    def _column(self, M, j):
        n = M.shape[0]
        return self._gather(M, _trees_arange(n), Array.from_list([j], "<i4")).reshape((n,))


# ------------------------------------------------------------------ Voting
# Reference: scikit-learn `sklearn/ensemble/_voting.py` (VotingClassifier
# :200 -- fit on the label-encoded y, 'hard' = weighted bincount argmax,
# 'soft' = weighted np.average of predict_proba --, VotingRegressor :500 --
# weighted np.average of predict). Members are cloned, never refitted in
# place; n_jobs and verbose are refused by name.
class VotingClassifier(_TreesWrapperBase):
    _estimator_type = "classifier"

    def __init__(self, estimators, *, voting="hard", weights=None, n_jobs=None, flatten_transform=True,
                 verbose=False):
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        if n_jobs is not None or verbose:
            _refuse("n_jobs/verbose", "the members fit one after another.")
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def _weights(self):
        m = len(self.estimators_)
        w = [1.0] * m if self.weights is None else [float(v) for v in self.weights]
        if len(w) != m:
            raise ValueError(f"weights has {len(w)} entries, there are {m} estimators")
        return w

    def fit(self, X, y, sample_weight=None):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        self.classes_, codes = encode_labels(y)
        self.n_classes_ = len(self.classes_)
        active = _trees_estimators(self.estimators)
        self.estimators_ = []
        for _, est in active:
            e = _trees_clone(est)
            e.fit(Xa, codes) if sample_weight is None else e.fit(Xa, codes, sample_weight=sample_weight)
            self.estimators_.append(e)
        self.named_estimators_ = dict((n, e) for (n, _), e in zip(active, self.estimators_))
        self.n_features_in_ = Xa.shape[1]
        self._fitted = True
        self._weights()
        return self

    def _soft(self, Xa):
        n, k = Xa.shape[0], self.n_classes_
        acc = zeros((n * k,), "<f8")
        w = self._weights()
        for e, wi in zip(self.estimators_, w):
            self._acc_cols(acc, e.predict_proba(Xa), _trees_sub_cols(e), n, k, wi)
        self._scale(acc, math.fsum(w))
        return acc

    def predict_proba(self, X):
        if self.voting == "hard":
            raise AttributeError("predict_proba is not available when voting='hard'")
        Xa = self._check_X(X)
        return self._soft(Xa).reshape((Xa.shape[0], self.n_classes_))

    def predict(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        if self.voting == "soft":
            acc = self._soft(Xa)
        else:
            acc = zeros((n * k,), "<f8")
            for e, wi in zip(self.estimators_, self._weights()):
                self._acc_votes(acc, as_i32_c(e.predict(Xa), ndim=1, name="codes")[0], n, k, wi)
        return decode_labels(self.classes_, self._argmax(acc, n, k))

    def transform(self, X):
        """soft: the members' probabilities side by side (n, m * k), float64;
        hard: the members' predicted codes (n, m), float64."""
        Xa = self._check_X(X)
        n, k, m = Xa.shape[0], self.n_classes_, len(self.estimators_)
        width = m * k if self.voting == "soft" else m
        out = zeros((n * width,), "<f8")
        rows = _trees_arange(n)
        for j, e in enumerate(self.estimators_):
            if self.voting == "soft":
                p = zeros((n * k,), "<f8")
                self._acc_cols(p, e.predict_proba(Xa), _trees_sub_cols(e), n, k)
                self._place(out, n, width, as_f32_c(p.reshape((n, k)), ndim=2, name="p")[0], rows, j * k)
            else:
                self._place(out, n, width, as_f32_c(e.predict(Xa), ndim=1, name="codes")[0], rows, j)
        return out.reshape((n, width))


class VotingRegressor(_TreesWrapperBase):
    _estimator_type = "regressor"

    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        if n_jobs is not None or verbose:
            _refuse("n_jobs/verbose", "the members fit one after another.")
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        active = _trees_estimators(self.estimators)
        self.estimators_ = []
        for _, est in active:
            e = _trees_clone(est)
            e.fit(Xa, y) if sample_weight is None else e.fit(Xa, y, sample_weight=sample_weight)
            self.estimators_.append(e)
        self.named_estimators_ = dict((n, e) for (n, _), e in zip(active, self.estimators_))
        self.n_features_in_ = Xa.shape[1]
        self._fitted = True
        return self

    def predict(self, X):
        Xa = self._check_X(X)
        n, m = Xa.shape[0], len(self.estimators_)
        w = [1.0] * m if self.weights is None else [float(v) for v in self.weights]
        if len(w) != m:
            raise ValueError(f"weights has {len(w)} entries, there are {m} estimators")
        acc = zeros((n,), "<f8")
        for e, wi in zip(self.estimators_, w):
            self._acc(acc, e.predict(Xa), n, wi)
        self._scale(acc, math.fsum(w))
        return acc

    def transform(self, X):
        Xa = self._check_X(X)
        n, m = Xa.shape[0], len(self.estimators_)
        out = zeros((n * m,), "<f8")
        rows = _trees_arange(n)
        for j, e in enumerate(self.estimators_):
            self._place(out, n, m, as_f32_c(e.predict(Xa), ndim=1, name="p")[0], rows, j)
        return out.reshape((n, m))


# ---------------------------------------------------------------- Stacking
# Reference: scikit-learn `sklearn/ensemble/_stacking.py` (_BaseStacking.fit
# :150 -- every member refitted on all rows, the meta features from
# cross_val_predict --, _concatenate_predictions :80 -- a binary
# predict_proba keeps only its second column --, stack_method 'auto' =
# predict_proba, decision_function, predict). cv is an int of unshuffled
# folds (StratifiedKFold for the classifier, KFold for the regressor).
# DEVIATION: the default final estimators are this library's
# LogisticRegression and Ridge (sklearn: LogisticRegression, RidgeCV).
class _StackingBase(_TreesWrapperBase):
    def __init__(self, estimators, final_estimator, cv, stack_method, n_jobs, passthrough, verbose):
        if n_jobs is not None or verbose:
            _refuse("n_jobs/verbose", "the members fit one after another.")
        if stack_method not in ("auto", "predict_proba", "decision_function", "predict"):
            raise ValueError("stack_method must be auto, predict_proba, decision_function or predict")
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose
        _trees_cv(cv)

    def _method(self, est):
        if self.stack_method != "auto":
            return self.stack_method
        for m in (("predict_proba", "decision_function", "predict") if self._estimator_type == "classifier"
                  else ("predict",)):
            if hasattr(est, m):
                return m
        return "predict"

    def _out(self, est, method, Xs):
        out = getattr(est, method)(Xs)
        if method == "predict_proba" and self._binary:
            n = Xs.shape[0]
            p = _trees_output_2d(out, n)
            return self._gather(p, _trees_arange(n), Array.from_list([1], "<i4"))
        return out

    def _widths(self, Xa, y_fit):
        return None

    def _fit_stack(self, X, y_fit, folds, final):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        active = _trees_estimators(self.estimators)
        self.estimators_ = []
        for _, est in active:
            e = _trees_clone(est)
            e.fit(Xa, y_fit(None))
            self.estimators_.append(e)
        self.stack_method_ = [self._method(e) for e in self.estimators_]
        widths = []
        for e, m in zip(self.estimators_, self.stack_method_):
            widths.append(_trees_output_2d(self._out(e, m, Xa[0:2]), 2).shape[1])
        width = sum(widths) + (d if self.passthrough else 0)
        meta = zeros((n * width,), "<f8")
        n_splits = max(folds) + 1
        cols = _trees_arange(d)
        for i in range(n_splits):
            tr, te = _trees_fold_rows(folds, i)
            col0 = 0
            for (_, est), m, w in zip(active, self.stack_method_, widths):
                e = _trees_clone(est)
                e.fit(self._gather(Xa, tr, cols), y_fit(tr))
                got = self._place(meta, n, width, self._out(e, m, self._gather(Xa, te, cols)), te, col0)
                if got != w:
                    raise ValueError("a member's output width changed between folds (a class missing from a fold)")
                col0 += w
        if self.passthrough:
            self._place(meta, n, width, Xa, _trees_arange(n), sum(widths))
        self._widths_ = widths
        self.named_estimators_ = dict((nm, e) for (nm, _), e in zip(active, self.estimators_))
        self.final_estimator_ = _trees_clone(final)
        self.final_estimator_.fit(as_f32_c(meta.reshape((n, width)), ndim=2, name="meta")[0], y_fit(None))
        self.n_features_in_ = d
        self._fitted = True
        return self

    def transform(self, X):
        Xa = self._check_X(X)
        n, d = Xa.shape
        width = sum(self._widths_) + (d if self.passthrough else 0)
        meta = zeros((n * width,), "<f8")
        rows, col0 = _trees_arange(n), 0
        for e, m in zip(self.estimators_, self.stack_method_):
            col0 += self._place(meta, n, width, self._out(e, m, Xa), rows, col0)
        if self.passthrough:
            self._place(meta, n, width, Xa, rows, col0)
        return meta.reshape((n, width))

    def _meta32(self, X):
        return as_f32_c(self.transform(X), ndim=2, name="meta")[0]


class StackingClassifier(_StackingBase):
    _estimator_type = "classifier"

    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method="auto", n_jobs=None,
                 passthrough=False, verbose=0):
        super().__init__(estimators, final_estimator, cv, stack_method, n_jobs, passthrough, verbose)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            _refuse("StackingClassifier sample_weight", "not carried in pass 1.")
        self.classes_, codes = encode_labels(y)
        self._binary = len(self.classes_) == 2
        folds = _trees_stratified_folds(codes.tolist(), _trees_cv(self.cv))
        final = self.final_estimator
        if final is None:
            from .linear_model import LogisticRegression
            final = LogisticRegression()
        return self._fit_stack(X, lambda rows: codes if rows is None else self._gather_codes(codes, rows),
                               folds, final)

    def predict(self, X):
        codes = as_i32_c(self.final_estimator_.predict(self._meta32(X)), ndim=1, name="codes")[0]
        return decode_labels(self.classes_, codes)

    def predict_proba(self, X):
        k = len(self.classes_)
        Xm = self._meta32(X)
        n = Xm.shape[0]
        acc = zeros((n * k,), "<f8")
        self._acc_cols(acc, self.final_estimator_.predict_proba(Xm), _trees_sub_cols(self.final_estimator_), n, k)
        return acc.reshape((n, k))


class StackingRegressor(_StackingBase):
    _estimator_type = "regressor"
    _binary = False

    def __init__(self, estimators, final_estimator=None, *, cv=None, n_jobs=None, passthrough=False, verbose=0):
        super().__init__(estimators, final_estimator, cv, "auto", n_jobs, passthrough, verbose)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            _refuse("StackingRegressor sample_weight", "not carried in pass 1.")
        y32, _ = as_f32_c(y, ndim=1, name="y")
        folds = _trees_kfolds(len(y32), _trees_cv(self.cv))
        final = self.final_estimator
        if final is None:
            from .linear_model import Ridge
            final = Ridge()
        return self._fit_stack(X, lambda rows: y32 if rows is None else self._gather_vec(y32, rows), folds, final)

    def predict(self, X):
        return self.final_estimator_.predict(self._meta32(X))


# ------------------------------------------------------------- MultiOutput
# Reference: scikit-learn `sklearn/multioutput.py` (_MultiOutputEstimator.fit
# :200, one clone per column of Y; predict stacks the columns;
# MultiOutputClassifier.predict_proba :500 returns a list). Y is a numeric
# 2-D buffer; a classifier's labels per column are encoded to codes.
class MultiOutputRegressor(_TreesWrapperBase):
    _estimator_type = "regressor"

    def __init__(self, estimator, *, n_jobs=None):
        if n_jobs is not None:
            _refuse("n_jobs", "the columns fit one after another.")
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, Y, sample_weight=None):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        Ya, _ = as_f32_c(Y, ndim=2, name="Y")
        if Ya.shape[0] != Xa.shape[0]:
            raise ValueError(f"Y has {Ya.shape[0]} rows, X has {Xa.shape[0]}")
        self.estimators_ = []
        for j in range(Ya.shape[1]):
            e = _trees_clone(self.estimator)
            yj = self._column(Ya, j)
            e.fit(Xa, yj) if sample_weight is None else e.fit(Xa, yj, sample_weight=sample_weight)
            self.estimators_.append(e)
        self.n_features_in_ = Xa.shape[1]
        self._fitted = True
        return self

    def predict(self, X):
        Xa = self._check_X(X)
        n, m = Xa.shape[0], len(self.estimators_)
        out = zeros((n * m,), "<f8")
        rows = _trees_arange(n)
        for j, e in enumerate(self.estimators_):
            self._place(out, n, m, as_f32_c(e.predict(Xa), ndim=1, name="p")[0], rows, j)
        return out.reshape((n, m))


class MultiOutputClassifier(_TreesWrapperBase):
    _estimator_type = "classifier"

    def __init__(self, estimator, *, n_jobs=None):
        if n_jobs is not None:
            _refuse("n_jobs", "the columns fit one after another.")
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, Y, sample_weight=None):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        rows = Y.tolist() if hasattr(Y, "tolist") else [list(r) for r in Y]
        if len(rows) != Xa.shape[0]:
            raise ValueError(f"Y has {len(rows)} rows, X has {Xa.shape[0]}")
        m = len(rows[0])
        self.estimators_, self.classes_ = [], []
        for j in range(m):
            classes, codes = encode_labels([r[j] for r in rows])
            e = _trees_clone(self.estimator)
            e.fit(Xa, codes) if sample_weight is None else e.fit(Xa, codes, sample_weight=sample_weight)
            self.estimators_.append(e)
            self.classes_.append(classes)
        self.n_features_in_ = Xa.shape[1]
        self._fitted = True
        return self

    def predict(self, X):
        """(n, n_outputs): an int64 or float64 Array for numeric labels."""
        Xa = self._check_X(X)
        cols = [decode_labels(c, as_i32_c(e.predict(Xa), ndim=1, name="codes")[0]).tolist()
                for e, c in zip(self.estimators_, self.classes_)]
        kind = "<i8" if all(isinstance(v, int) for col in cols for v in col[:1]) else "<f8"
        return Array.from_list([list(r) for r in zip(*cols)], kind)

    def predict_proba(self, X):
        """A list, one (n, n_classes_j) float64 Array per output."""
        Xa = self._check_X(X)
        n = Xa.shape[0]
        out = []
        for e, c in zip(self.estimators_, self.classes_):
            acc = zeros((n * len(c),), "<f8")
            self._acc_cols(acc, e.predict_proba(Xa), _trees_sub_cols(e), n, len(c))
            out.append(acc.reshape((n, len(c))))
        return out


# --------------------------------------------------------------- OneVsRest
# Reference: scikit-learn `sklearn/multiclass.py` OneVsRestClassifier (fit
# :330, one binary clone per class -- a single one for two classes --;
# predict :420 argmax of decision_function, else of predict_proba[:, 1];
# predict_proba :470 each class's positive probability, rows normalised in
# the multiclass case). DEVIATION: a row whose scores sum to 0 is uniform,
# where sklearn divides 0 / 0.
class OneVsRestClassifier(_TreesWrapperBase):
    _estimator_type = "classifier"

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        if n_jobs is not None or verbose:
            _refuse("n_jobs/verbose", "the classes fit one after another.")
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        self.classes_, codes = encode_labels(y)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("y has fewer than 2 classes")
        cl = codes.tolist()
        targets = [codes] if k == 2 else [Array.from_list([1 if c == j else 0 for c in cl], "<i4") for j in range(k)]
        self.estimators_ = []
        for t in targets:
            e = _trees_clone(self.estimator)
            e.fit(Xa, t)
            self.estimators_.append(e)
        self.n_features_in_ = Xa.shape[1]
        self._fitted = True
        return self

    def _positive(self, e, Xa):
        """The estimator's score for class 1, float32 (n,): decision_function
        when it has one, else predict_proba's column for code 1."""
        n = Xa.shape[0]
        if hasattr(e, "decision_function"):
            return as_f32_c(e.decision_function(Xa), ndim=1, name="score")[0]
        p = _trees_output_2d(e.predict_proba(Xa), n)
        subs = [int(c) for c in e.classes_]
        if 1 not in subs:
            return as_f32_c(zeros((n,), "<f8"), ndim=1, name="score")[0]
        return self._gather(p, _trees_arange(n), Array.from_list([subs.index(1)], "<i4")).reshape((n,))

    def _proba_positive(self, e, Xa):
        n = Xa.shape[0]
        p = _trees_output_2d(e.predict_proba(Xa), n)
        subs = [int(c) for c in e.classes_]
        if 1 not in subs:
            return as_f32_c(zeros((n,), "<f8"), ndim=1, name="p")[0]
        return self._gather(p, _trees_arange(n), Array.from_list([subs.index(1)], "<i4")).reshape((n,))

    def predict(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], len(self.classes_)
        if k == 2:
            codes = as_i32_c(self.estimators_[0].predict(Xa), ndim=1, name="codes")[0]
            return decode_labels(self.classes_, codes)
        acc = zeros((n * k,), "<f8")
        rows = _trees_arange(n)
        for j, e in enumerate(self.estimators_):
            self._place(acc, n, k, self._positive(e, Xa), rows, j)
        return decode_labels(self.classes_, self._argmax(acc, n, k))

    def predict_proba(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], len(self.classes_)
        acc = zeros((n * k,), "<f8")
        rows = _trees_arange(n)
        if k == 2:
            p = self._proba_positive(self.estimators_[0], Xa).tolist()
            return Array.from_list([[1.0 - v, v] for v in p], "<f8")
        for j, e in enumerate(self.estimators_):
            self._place(acc, n, k, self._proba_positive(e, Xa), rows, j)
        self._bind().x_trees_normalize_rows(addr(acc, name="proba"), [n, k])
        return acc.reshape((n, k))


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
class CalibratedClassifierCV(_TreesWrapperBase):
    _estimator_type = "classifier"

    def __init__(self, estimator=None, *, method="sigmoid", cv=None, n_jobs=None, ensemble=True):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        if n_jobs is not None:
            _refuse("n_jobs", "the folds fit one after another.")
        if ensemble not in (True, False):
            _refuse(f"ensemble={ensemble!r}", "True or False.")
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        _trees_cv(cv)

    def _scores(self, e, Xa):
        """(n, c) float64 scores: c = 1 when binary, else one column per class
        (columns in the member's class-code order, mapped to all k)."""
        n, k = Xa.shape[0], len(self.classes_)
        if hasattr(e, "decision_function"):
            out = e.decision_function(Xa)
        else:
            out = e.predict_proba(Xa)
        blk = _trees_output_2d(out, n)
        subs = [int(c) for c in e.classes_]
        if k == 2:
            if blk.shape[1] == 1:
                col = blk
            else:
                col = self._gather(blk, _trees_arange(n), Array.from_list([subs.index(1)], "<i4"))
            acc = zeros((n,), "<f8")
            self._acc(acc, col.reshape((n,)), n)
            return acc.reshape((n, 1))
        acc = zeros((n * k,), "<f8")
        self._acc_cols(acc, blk, Array.from_list(subs, "<i4"), n, k)
        return acc.reshape((n, k))

    def _fit_calibrators(self, S, codes):
        n, c = S.shape
        b = self._bind()
        cl = codes.tolist()
        cals = []
        for j in range(c):
            cls = 1 if c == 1 else j
            f = self._column64(S, j)
            yj = Array.from_list([1 if v == cls else 0 for v in cl], "<i4")
            if self.method == "sigmoid":
                ab = zeros((2,), "<f8")
                b.x_trees_platt_fit(addr_ro(f, name="f"), addr_ro(yj, name="y"), addr(ab, name="ab"), [n])
                cals.append(("sigmoid", tuple(ab.tolist())))
            else:
                y64 = Array.from_list([float(v) for v in yj.tolist()], "<f8")
                kx, ky = empty((n,), "<f8"), empty((n,), "<f8")
                m = int(b.x_trees_isotonic_fit(addr_ro(f, name="x"), addr_ro(y64, name="y"), addr(kx, name="kx"),
                                               addr(ky, name="ky"), [n]))
                cals.append(("isotonic", (kx[0:m], ky[0:m], m)))
        return cals

    def _column64(self, S, j):
        n, c = S.shape
        if c == 1:
            return S.reshape((n,))
        v = S.tolist()
        return Array.from_list([row[j] for row in v], "<f8")

    def _calibrated(self, e, cals, Xa):
        n, k = Xa.shape[0], len(self.classes_)
        S = self._scores(e, Xa)
        b = self._bind()
        cols = []
        for j, (kind, par) in enumerate(cals):
            f = self._column64(S, j)
            out = empty((n,), "<f8")
            if kind == "sigmoid":
                b.x_trees_platt_apply(addr_ro(f, name="f"), addr(out, name="p"), [n, par[0], par[1]])
            else:
                kx, ky, m = par
                b.x_trees_isotonic_predict(addr_ro(kx, name="kx"), addr_ro(ky, name="ky"), addr_ro(f, name="t"),
                                           addr(out, name="p"), [m, n])
            cols.append(out.tolist())
        if k == 2:
            p = cols[0]
            return Array.from_list([v for x in p for v in (1.0 - x, x)], "<f8")
        acc = Array.from_list([cols[j][i] for i in range(n) for j in range(k)], "<f8")
        b.x_trees_normalize_rows(addr(acc, name="proba"), [n, k])
        return acc

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            _refuse("CalibratedClassifierCV sample_weight", "not carried in pass 1.")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        self.classes_, codes = encode_labels(y)
        if len(self.classes_) < 2:
            raise ValueError("y has fewer than 2 classes")
        base = self.estimator
        if base is None:
            from .svm import LinearSVC
            base = LinearSVC()
        folds = _trees_stratified_folds(codes.tolist(), _trees_cv(self.cv))
        cols = _trees_arange(d)
        self.calibrated_classifiers_ = []
        if self.ensemble:
            for i in range(max(folds) + 1):
                tr, te = _trees_fold_rows(folds, i)
                e = _trees_clone(base)
                e.fit(self._gather(Xa, tr, cols), self._gather_codes(codes, tr))
                Xte = self._gather(Xa, te, cols)
                cals = self._fit_calibrators(self._scores(e, Xte), self._gather_codes(codes, te))
                self.calibrated_classifiers_.append((e, cals))
        else:
            k = len(self.classes_)
            c = 1 if k == 2 else k
            S = zeros((n * c,), "<f8")
            for i in range(max(folds) + 1):
                tr, te = _trees_fold_rows(folds, i)
                e = _trees_clone(base)
                e.fit(self._gather(Xa, tr, cols), self._gather_codes(codes, tr))
                self._place(S, n, c, as_f32_c(self._scores(e, self._gather(Xa, te, cols)), ndim=2,
                                              name="scores")[0], te, 0)
            e = _trees_clone(base)
            e.fit(Xa, codes)
            self.calibrated_classifiers_.append((e, self._fit_calibrators(S.reshape((n, c)), codes)))
        self.n_features_in_ = d
        self._fitted = True
        return self

    def predict_proba(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], len(self.classes_)
        acc = zeros((n * k,), "<f8")
        for e, cals in self.calibrated_classifiers_:
            p = self._calibrated(e, cals, Xa)
            self._acc(acc, p, n * k)
        self._scale(acc, len(self.calibrated_classifiers_))
        return acc.reshape((n, k))

    def predict(self, X):
        p = self.predict_proba(X)
        return decode_labels(self.classes_, self._argmax(p, p.shape[0], len(self.classes_)))


# -------------------------------------------------------------------- SHAP
# TreeExplainer: the `shap` package's exact path-dependent TreeSHAP
# (`shap/explainers/_tree.py` -> `shap/cext/tree_shap.h`), restated in
# xtrees/shap.mojo over this library's flat forests (RandomForest*,
# ExtraTrees*, DecisionTree*) and DART. DEVIATION: the node cover is the
# count of BACKGROUND rows (`data`, required) reaching each node, since the
# flat forest stores no instance counts. KernelExplainer and
# PermutationExplainer: cuML's `explainer/kernel_shap.cu` and
# `permutation_shap.cu` build the coalition datasets (xtrees/shap.mojo
# `mask_expand`); the sampling and the solve follow `shap`'s
# KernelExplainer (`_kernel.py`: full enumeration of the small coalition
# sizes, then weighted sampling of the rest, the efficiency-constrained
# weighted least squares) and PermutationExplainer (`_permutation.py`:
# forward then backward passes over each permutation). DEVIATIONS: draws
# come from the lane's counter RNG; KernelExplainer carries no l1 feature
# selection (`l1_reg` is refused) and treats every feature as varying.
def _trees_forest_arrays(est):
    """(offsets, colid, quesval, left, leaves, k, scale) of a fitted flat
    forest, or None."""
    if not hasattr(est, "_offsets"):
        return None
    k = int(getattr(est, "_num_outputs", 1))
    return (est._offsets, est._colid, est._quesval, est._left_child, est._leaves, k,
            1.0 / int(est._n_trees))


class TreeExplainer(_TreesEnsembleBase):
    """Exact TreeSHAP for this library's forests and DART models.
    `shap_values(X)` is (n, d) for one output, else (n, d, k), float64;
    `expected_value` is a float or a float64 Array of k."""

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
        self._parts = []   # (arrays tuple, k, scale, cover)
        init = 0.0
        if isinstance(model, _DARTBase):
            if not hasattr(model, "trees_"):
                raise RuntimeError("the model is not fitted yet")
            init = float(model.init_score_)
            for tree, values, coef in zip(model.trees_, model.tree_values_, model.tree_coefs_):
                self._add_part((tree._offsets, tree._colid, tree._quesval, tree._left_child, values), 1, float(coef))
            self.n_outputs_ = 1
        else:
            fa = _trees_forest_arrays(model)
            if fa is None:
                _refuse(f"TreeExplainer over {type(model).__name__}", "the explainer reads the flat forests"
                        " (RandomForest*, ExtraTrees*, DecisionTree*) and DART models.")
            self._add_part(fa[:5], fa[5], fa[6])
            self.n_outputs_ = fa[5]
        k = self.n_outputs_
        ev = full((k,), init, "<f8")
        for arrays, kk, scale, cover in self._parts:
            n_trees = len(arrays[0]) - 1
            self._bind().x_trees_expected_value(addr_ro(arrays[0], name="offsets"), addr_ro(arrays[3], name="left"),
                                                addr_ro(arrays[4], name="leaves"), addr_ro(cover, name="cover"),
                                                addr(ev, name="ev"), [n_trees, kk, scale])
        self.expected_value = ev.tolist()[0] if k == 1 else ev
        self.n_features_in_ = bg.shape[1]

    def _add_part(self, arrays, k, scale):
        bg = self._bg
        n_nodes = int(arrays[0].tolist()[-1])
        cover = zeros((n_nodes,), "<f8")
        self._bind().x_trees_node_cover(addr_ro(arrays[0], name="offsets"), addr_ro(arrays[1], name="colid"),
                                        addr_ro(arrays[2], name="quesval"), addr_ro(arrays[3], name="left"),
                                        addr_ro(bg, name="data"), addr(cover, name="cover"),
                                        [bg.shape[0], bg.shape[1], len(arrays[0]) - 1])
        self._parts.append((arrays, k, scale, cover))

    def shap_values(self, X):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        if d != self.n_features_in_:
            raise ValueError(f"X has {d} features, data has {self.n_features_in_}")
        k = self.n_outputs_
        phi = zeros((n * d * k,), "<f8")
        for arrays, kk, scale, cover in self._parts:
            forest = [addr_ro(a, name="forest") for a in arrays]
            self._bind().x_trees_tree_shap(forest, addr_ro(cover, name="cover"), addr_ro(Xa, name="X"),
                                           addr(phi, name="phi"), [n, d, len(arrays[0]) - 1, kk, scale])
        return phi.reshape((n, d)) if k == 1 else phi.reshape((n, d, k))


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

    def _eval(self, X):
        """The model output on X as a float32 (n, k) Array."""
        out = self._f(X)
        n = X.shape[0]
        return _trees_output_2d(out, n)

    def _coalitions(self, x_row, masks_list):
        """Mean model output over the background for each coalition mask:
        float64 (m, k)."""
        bg = self._bg
        nb, d = bg.shape
        m = len(masks_list)
        masks = Array.from_list([v for mk in masks_list for v in mk], "<i4")
        syn = empty((m * nb * d,), "<f4")
        b = self._bind()
        b.x_trees_mask_expand(addr_ro(x_row, name="x"), addr_ro(bg, name="data"), addr_ro(masks, name="masks"),
                              addr(syn, name="synthetic"), [nb, d, m])
        out = self._eval(syn.reshape((m * nb, d)))
        k = out.shape[1]
        ey = empty((m * k,), "<f8")
        b.x_trees_block_mean(addr_ro(out, name="y"), addr(ey, name="ey"), [m, nb, k])
        return masks, ey

    def _check(self, X):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, data has {self.n_features_in_}")
        return Xa

    def _shape(self, rows, n, d):
        k = self.n_outputs_
        flat = Array.from_list([v for r in rows for v in r], "<f8")
        return flat.reshape((n, d)) if k == 1 else flat.reshape((n, d, k))


class KernelExplainer(_AgnosticExplainer):
    """Kernel SHAP (shap `KernelExplainer`, cuML `kernel_shap.cu`) over a
    mojolearn estimator or a callable. `shap_values(X, nsamples="auto")`."""

    def __init__(self, model, data, *, link="identity", random_state=None):
        if link != "identity":
            _refuse(f"link={link!r}", "only the identity link is carried.")
        self.link = link
        super().__init__(model, data, random_state)

    @staticmethod
    def _binom(n, r):
        from math import comb
        return comb(n, r)

    def _masks(self, M, nsamples, seed, row):
        """shap `KernelExplainer.explain`'s coalition schedule: (masks, weights)."""
        import itertools
        masks, weights = [], []
        num_subset_sizes = (M - 1 + 1) // 2 if M > 1 else 0
        num_paired = (M - 1) // 2
        wv = [(M - 1.0) / (i * (M - i)) for i in range(1, num_subset_sizes + 1)]
        for i in range(num_paired):
            wv[i] *= 2
        tot = math.fsum(wv)
        wv = [w / tot for w in wv]
        num_full = 0
        left = nsamples
        rem = list(wv)
        for size in range(1, num_subset_sizes + 1):
            nsub = self._binom(M, size) * (2 if size <= num_paired else 1)
            if left * rem[size - 1] / nsub >= 1.0 - 1e-8:
                num_full += 1
                left -= nsub
                if rem[size - 1] < 1.0:
                    r0 = rem[size - 1]
                    rem = [v / (1 - r0) for v in rem]
                w = wv[size - 1] / self._binom(M, size)
                if size <= num_paired:
                    w /= 2.0
                for inds in itertools.combinations(range(M), size):
                    mk = [0] * M
                    for i in inds:
                        mk[i] = 1
                    masks.append(mk)
                    weights.append(w)
                    if size <= num_paired:
                        masks.append([1 - v for v in mk])
                        weights.append(w)
            else:
                break
        nfixed = len(masks)
        samples_left = nsamples - nfixed
        if num_full != num_subset_sizes and samples_left > 0:
            rw = list(wv)
            for i in range(num_paired):
                rw[i] /= 2
            rw = rw[num_full:]
            t = math.fsum(rw)
            rw = [v / t for v in rw]
            cdf, run = [], 0.0
            for v in rw:
                run += v
                cdf.append(run)
            n_draw = 4 * samples_left
            u = empty((n_draw * (1 + M),), "<f8")
            self._bind().x_trees_uniform(addr(u, name="u"), [n_draw * (1 + M), seed, row])
            uv = u.tolist()
            used = {}
            pos = 0
            while samples_left > 0 and pos < n_draw:
                base = pos * (1 + M)
                c = uv[base] * cdf[-1]
                ind = next((i for i, v in enumerate(cdf) if c < v), len(cdf) - 1)
                pos += 1
                size = ind + num_full + 1
                perm = list(range(M))
                for i in range(M - 1, 0, -1):
                    j = int(uv[base + 1 + i] * (i + 1))
                    perm[i], perm[j] = perm[j], perm[i]
                mk = [0] * M
                for i in perm[:size]:
                    mk[i] = 1
                key = tuple(mk)
                new = key not in used
                if new:
                    used[key] = len(masks)
                    samples_left -= 1
                    masks.append(mk)
                    weights.append(1.0)
                else:
                    weights[used[key]] += 1.0
                if samples_left > 0 and size <= num_paired:
                    if new:
                        samples_left -= 1
                        masks.append([1 - v for v in mk])
                        weights.append(1.0)
                    else:
                        weights[used[key] + 1] += 1.0
            weight_left = math.fsum(wv[num_full:])
            s = math.fsum(weights[nfixed:])
            if s > 0:
                weights[nfixed:] = [w * (weight_left / s) for w in weights[nfixed:]]
        return masks, weights

    def shap_values(self, X, nsamples="auto", l1_reg="auto"):
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
        rows = []
        cols = _trees_arange(d)
        for i in range(n):
            x_row = self._gather(Xa, Array.from_list([i], "<i4"), cols).reshape((d,))
            fx = zeros((k,), "<f8")
            fx32 = self._eval(x_row.reshape((1, d)))   # held: the call reads its address
            b.x_trees_block_mean(addr_ro(fx32, name="fx"), addr(fx, name="fx"), [1, 1, k])
            phi = zeros((d * k,), "<f8")
            if M == 1 or ns < 1:
                phi = Array.from_list([fx.tolist()[j] - self._fnull.tolist()[j] for j in range(k)], "<f8")
            else:
                mlist, wlist = self._masks(M, ns, seed, i)
                masks, ey = self._coalitions(x_row, mlist)
                w = Array.from_list(wlist, "<f8")
                b.x_trees_kernel_solve(addr_ro(masks, name="masks"), addr_ro(w, name="w"), addr_ro(ey, name="ey"),
                                       addr_ro(fx, name="fx"), addr_ro(self._fnull, name="fnull"),
                                       addr(phi, name="phi"), [len(mlist), d, k])
            rows.append(phi.tolist())
        return self._shape(rows, n, d)


class PermutationExplainer(_AgnosticExplainer):
    """Permutation SHAP (shap `PermutationExplainer`, cuML
    `permutation_shap.cu`): each permutation adds the features one by one
    (forward) then removes them (backward), each marginal averaged over the
    background. `shap_values(X, npermutations=10)`."""

    def __init__(self, model, data, *, random_state=None):
        super().__init__(model, data, random_state)

    def shap_values(self, X, npermutations=10):
        Xa = self._check(X)
        n, d = Xa.shape
        k = self.n_outputs_
        seed = _trees_seed(self.random_state)
        rows = []
        cols = _trees_arange(d)
        for i in range(n):
            x_row = self._gather(Xa, Array.from_list([i], "<i4"), cols).reshape((d,))
            u = empty((max(1, npermutations * d),), "<f8")
            self._bind().x_trees_uniform(addr(u, name="u"), [npermutations * d, seed, i])
            uv = u.tolist()
            masks, perms = [], []
            for p in range(int(npermutations)):
                perm = list(range(d))
                for a in range(d - 1, 0, -1):
                    j = int(uv[p * d + a] * (a + 1))
                    perm[a], perm[j] = perm[j], perm[a]
                perms.append(perm)
                cur = [0] * d
                masks.append(list(cur))
                for f in perm:
                    cur[f] = 1
                    masks.append(list(cur))
                for f in perm:
                    cur[f] = 0
                    masks.append(list(cur))
            _, ey = self._coalitions(x_row, masks)
            e = ey.tolist()
            val = [0.0] * (d * k)
            step = 2 * d + 1
            for p, perm in enumerate(perms):
                o = p * step
                for jj, f in enumerate(perm):
                    for c in range(k):
                        val[f * k + c] += e[(o + jj + 1) * k + c] - e[(o + jj) * k + c]
                for jj, f in enumerate(perm):
                    for c in range(k):
                        val[f * k + c] += e[(o + d + jj) * k + c] - e[(o + d + jj + 1) * k + c]
            den = 2.0 * npermutations
            rows.append([v / den for v in val])
        return self._shape(rows, n, d)
