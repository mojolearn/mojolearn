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

    def _check_X(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, fit saw {self.n_features_in_}")
        return Xa


class AdaBoostClassifier(_AdaBoostBase):
    """sklearn's `AdaBoostClassifier` (SAMME) over any mojolearn classifier,
    default `DecisionTreeClassifier(max_depth=1)`. DEVIATION: each member fits
    a WEIGHTED BOOTSTRAP of the rows drawn from the boosting weights (the
    resampling form of AdaBoost, as sklearn's R2 regressor does), where
    sklearn passes the weights as `sample_weight`: the forest's weighted
    objective has no CPU restatement, and one spelling serves every column.
    The error, the estimator weight and the reweighting are SAMME's, on the
    full training rows."""
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
        cols = _trees_arange(Xa.shape[1])
        self.estimators_, self.estimator_weights_, self.estimator_errors_ = [], [], []
        b = self._bind()
        m = int(self.n_estimators)
        for it in range(m):
            rows = empty((n,), "<i4")
            b.x_trees_weighted_sample(addr_ro(w, name="w"), addr(rows, name="rows"), [n, n, seed, it])
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, it))
            est.fit(self._gather(Xa, rows, cols), self._gather_codes(codes, rows))
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
