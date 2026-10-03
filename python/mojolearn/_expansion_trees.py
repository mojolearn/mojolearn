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
NOT YET TRUE, owed to the trees family (python_work_audit trees items 3, 9,
10), and NOT covered by any ledger row: Kernel/PermutationExplainer's per-row
coalition weights and marginals, Bagging's oob R^2 (`_portable_math.fsum` over
n) and AdaBoost's sample_weight normalization are still binary64 Python over
rows. CalibratedClassifierCV's per-class epilogue moved to xtrees on lane
py-misc-prep (strided platt/isotonic apply, `complement_pairs`, native class
columns and 0/1 targets); `_CAL_NATIVE = False` is its Python reference.
"""
import numbers
import os

from . import _portable_math as math
from . import _mojolearn_rf, _mojolearn_x_trees  # noqa: F401  the bindings this door resolves; name NO other (lane_select counts > 3 as a registry)
from ._array import Array
from ._buffer import _materialize, addr, addr_ro, all_finite, as_f32_c, as_f32_colmajor, as_f64_c, as_i32_c, empty, frombytes, full, zeros
from ._buffer import memory_at
from ._labels import decode_labels, encode_labels, is_bool
from ._mode import NumericModeMixin
from ._forest_protocol import (forest_estimator, _forest_fit_arrays, _forest_fit_function,
                               forest_data_session_choice, open_forest_data_session)
from .extratrees import ExtraTreesRegressor, ExtraTreesClassifier as _ET_CLS
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
def _trees_x_bind(est=None):
    from . import _backend
    return _backend.binding("_mojolearn_x_trees", getattr(est, "numeric_mode", None))


def _trees_f32_weights(values, n, name, est=None):
    """`values` (buffer or sequence) as a float32 Array of n entries,
    refusing non-finite and negative entries and an all-zero vector (a
    sample weight). One native pass (`x_trees_check_weights_f32`); the
    refusals and their wording are the per-element Python loop's."""
    arr, _ = as_f32_c(values, ndim=1, name=name)
    if len(arr) != n:
        raise ValueError(f"{name} has {len(arr)} entries, X has {n} rows")
    status = int(_trees_x_bind(est).x_trees_check_weights_f32(addr_ro(arr, name=name), [n]))
    if status == 1:
        raise ValueError(f"{name} must be finite and nonnegative")
    if status == 2:
        raise ValueError(f"{name} must have a positive total")
    return arr


def _trees_colmajor(X, est):
    """X as a column-major float32 Array: a float32 F-order input is borrowed
    as is (no copy); anything else goes through the lane's own threaded
    transpose, so the weighted fit (which takes the column-major layout)
    needs no base binding helper on a CPU-only install."""
    Xm, _ = _materialize(X, "X")
    if Xm.ndim == 2 and Xm.dtype == "<f4" and Xm.order == "F" and not Xm._both_orders():
        return as_f32_colmajor(Xm, name="X")[0]
    Xa, _ = as_f32_c(Xm, ndim=2, name="X")
    n, d = Xa.shape
    out = empty((d, n), "<f4")
    _trees_x_bind(est).x_trees_transpose_f32(addr_ro(Xa, name="X"), addr(out, name="X^T"), [n, d])
    return Array._view_of(out, (n, d), order="F")


def _trees_weighted_rows(sample_weight, class_weight, classes, codes, est=None):
    """Per-row float32 weights: sample_weight times the class_weight row
    weight. Each product of two float32 values is exact in binary64, so the
    one rounding to float32 is the correctly rounded float32 product on every
    host (`x_trees_mul_f32`)."""
    n = len(codes)
    sw = _trees_f32_weights(sample_weight, n, "sample_weight", est)
    if class_weight is None:
        return sw
    cw, _ = as_f32_c(_class_weight_rows(class_weight, classes, codes.tolist()), ndim=1, name="class_weight rows")
    out = empty((n,), "<f4")
    _trees_x_bind(est).x_trees_mul_f32(addr_ro(sw, name="sample_weight"), addr_ro(cw, name="class_weight"),
                                       addr(out, name="weights"), [n])
    return out


def _trees_fsum_f32(b, arr):
    """`math.fsum(arr.tolist())` of a float32 Array, bit for bit: the binding
    returns the EXACT sum as integer places of 2^-149 and this rounds it once
    with `_portable_math`'s own rounding (`_scaled_integer`, the last step of
    its `fsum`), so no value can differ. A NaN or infinity takes the Python
    fsum, which owns those cases. DARTRegressor/Classifier spent 1.3 s of a
    2.6 s 20-tree fit in the Python fsum at 1,000,000 rows (M3 Ultra,
    2026-09-28, trees-apple profile)."""
    from ._portable_math import _scaled_integer
    arr, _ = as_f32_c(arr, ndim=1, name="y")
    limbs = b.x_trees_exact_sum_f32(addr_ro(arr, name="y"), [arr.size])
    if limbs is None:
        return math.fsum(arr.tolist())
    total = 0
    for i, v in enumerate(limbs):
        total += int(v) << (32 * i)
    return _scaled_integer(total, -149)


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


def _dt_random_fit(est, cls, X, y, **extra):
    """splitter='random': one ExtraTrees tree; its flat arrays become this
    estimator's, and predict delegates to it."""
    inner = cls(n_estimators=1, bootstrap=False, max_features=1.0 if est.max_features is None else est.max_features,
                max_depth=est.max_depth, min_samples_split=est.min_samples_split,
                min_samples_leaf=est.min_samples_leaf, max_leaf_nodes=est.max_leaf_nodes,
                min_impurity_decrease=est.min_impurity_decrease, criterion=est.criterion,
                random_state=0 if est.random_state is None else est.random_state,
                numeric_mode=est.numeric_mode, **extra).fit(X, y)
    est._random_inner = inner
    for name in ("_offsets", "_colid", "_quesval", "_left_child", "_leaves", "_n_trees", "_num_outputs",
                 "n_features_in_"):
        setattr(est, name, getattr(inner, name))
    if hasattr(inner, "classes_"):
        est.classes_ = inner.classes_
        est.n_classes_ = len(inner.classes_)
    return est


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
        if self.splitter == "random":
            if sample_weight is not None or self.class_weight is not None:
                _refuse("splitter='random' with weights", "the ExtraTrees builder takes no row weights.")
            self._refresh_config()
            return _dt_random_fit(self, _ET_CLS, X, y)
        if sample_weight is None:
            return self._fit_with_tree_start(X, y)
        return self._fit_weighted(_trees_colmajor(X, self), y, sample_weight)

    def _fit_weighted(self, Xcm, y, sample_weight, x_finite=False, encoded=None, session=None):
        """The weighted fit on a column-major float32 X. AdaBoost calls it
        with ONE transposed, once-checked X for all of its members
        (`x_finite=True` skips the per-member finite scan of the same
        bytes) and, trees-apple2, ONE `encode_labels(y)` of the same codes
        (`encoded`, the pair this call would compute); every input the fit
        entry sees is what `fit` hands it. trees-apple3: `session` is a
        `ForestDataSession` that already holds this X on the device, so the
        member stages only its labels and weights."""
        self._refresh_config()
        self._capture_fit_mode()
        self.classes_, y32 = encoded if encoded is not None else encode_labels(y)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("y has fewer than 2 classes")
        weights = _trees_weighted_rows(sample_weight, self.class_weight, self.classes_, y32, self)
        binding = self._bind("_mojolearn_rf")
        if session is not None:
            if tuple(session.shape) != tuple(Xcm.shape):
                raise ValueError("the forest data session holds another X")

            def session_fit(x_addr, y_addr, params, criterion):
                return session.fit_classifier_weighted(y32, params, criterion, weights)
            return self._fit_colmajor_checked(Xcm, y32, session_fit)
        weighted_fit = _forest_fit_function(binding, "rf_classifier_fit_weighted")

        def fit_fn(x_addr, y_addr, params, criterion):
            return weighted_fit(x_addr, y_addr, params, criterion, addr_ro(weights, name="weights"))
        if x_finite:
            return self._fit_colmajor_checked(Xcm, y32, fit_fn)
        return self._fit_arrays(Xcm, y32, self.n_classes_, fit_fn)

    def _fit_colmajor_checked(self, Xcm, y32, fit_fn):
        """`RandomForestClassifier._fit_arrays` on a column-major float32 X
        whose finite scan the caller already ran on these exact bytes: the
        same parameters, the same entry, the same unpacking, minus that one
        scan."""
        n_rows, n_features = Xcm.shape
        if len(y32) != n_rows:
            raise ValueError(f"y has {len(y32)} rows, X has {n_rows}")
        params = self._fit_params(n_rows, n_features, self.n_classes_)
        out = fit_fn(addr_ro(Xcm, name="X"), addr_ro(y32, name="y"), params, self._cfg["criterion"])
        (self._offsets, self._colid, self._quesval, self._left_child,
         self._leaves, meta) = _forest_fit_arrays(out)
        self.n_features_in_ = int(n_features)
        self._n_trees = int(meta[0])
        self._num_outputs = int(meta[1])
        return self

    def get_depth(self):
        return _trees_depth(self)

    def get_n_leaves(self):
        return _trees_n_leaves(self)

    def predict_proba(self, X):
        inner = getattr(self, "_random_inner", None)
        return inner.predict_proba(X) if inner is not None else super().predict_proba(X)

    def predict(self, X):
        inner = getattr(self, "_random_inner", None)
        return inner.predict(X) if inner is not None else super().predict(X)

    def save(self, path):
        if getattr(self, "_random_inner", None) is not None:
            raise NotImplementedError("a splitter='random' tree saves through ExtraTrees; not carried yet")
        return super().save(path)


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
        rf_leaves = _dt_common(splitter, max_features, max_leaf_nodes)
        super().__init__(
            n_estimators=1, criterion=criterion, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
            max_leaf_nodes=rf_leaves, min_impurity_decrease=min_impurity_decrease,
            bootstrap=False, random_state=random_state, ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst, n_bins=n_bins, n_streams=1, device=device,
            inference_engine=inference_engine,
        )
        self.splitter = splitter

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            _refuse("DecisionTreeRegressor sample_weight", "the regressor fit entry"
                    " carries no row weights.")
        if self.splitter == "random":
            self._refresh_config()
            return _dt_random_fit(self, ExtraTreesRegressor, X, y)
        return self._fit_with_tree_start(X, y)

    def get_depth(self):
        return _trees_depth(self)

    def get_n_leaves(self):
        return _trees_n_leaves(self)

    def predict(self, X):
        inner = getattr(self, "_random_inner", None)
        return inner.predict(X) if inner is not None else super().predict(X)

    def save(self, path):
        if getattr(self, "_random_inner", None) is not None:
            raise NotImplementedError("a splitter='random' tree saves through ExtraTrees; not carried yet")
        return super().save(path)


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


def _trees_member_native(member):
    """The forest binding a clone of `member` resolves."""
    probe = _trees_clone(member)
    probe._capture_fit_mode()
    return probe._bind("_mojolearn_rf")


def _trees_member_session(member, X, row_major, x_finite=False, default="0"):
    """The data session (trees-apple3) for the member fits of one boosted
    ensemble, on the forest binding a clone of `member` resolves, or None
    (the choice is off, or the binary has no session entry). The session
    stands in for every member's finite scan of X, so X is scanned here
    unless the caller already did (`x_finite`)."""
    if forest_data_session_choice(None, default) is None or not hasattr(member, "_capture_fit_mode"):
        return None
    probe = _trees_clone(member)
    probe._capture_fit_mode()
    native = probe._bind("_mojolearn_rf")
    if not callable(getattr(native, "rf_data_session_open", None)):
        return None
    if not x_finite and not all_finite(X):
        raise ValueError("X contains NaN or infinity; the forest has no missing-value arm")
    return open_forest_data_session(native, X, row_major=row_major, mode=probe._effective_mode(),
                                    default=default)


def _class_major_fill(inits, n):
    """K * n float64, class c's block all `inits[c]` (class-major), built
    from one native fill per class, not a Python loop over the rows."""
    if len(inits) == 1:
        return full((n,), inits[0], "<f8")
    raw = b"".join(full((n,), float(v), "<f8").tobytes() for v in inits)
    return frombytes(raw, "<f8", (len(inits) * n,))


def _trees_arange(n):
    return Array.from_list(list(range(n)), "<i4")


# lane/apple-fast-trees-ensembles (2026-10-02): the Apple FAST switches of
# this module are BUILD-TIME defines of the x_trees binding (`-D
# MOJOLEARN_TE_<NAME>`; xtrees/api.mojo `x_trees_fast_switches` is the bit
# set the FAST + Apple build fills, 0 in every other build), read through
# the binding once per fit: no env read. IDENTICAL (the default) runs the
# code above and below them unchanged. Each defaults off:
#   MOJOLEARN_TE_NATIVE_SPLITS  the cv fold bookkeeping of Stacking /
#     CalibratedClassifierCV, OneVsRest's targets and MultiOutputClassifier's
#     label columns leave their Python row loops (`_trees_native_folds`,
#     `_trees_native_glue`);
#   MOJOLEARN_TE_ADA_SESSION  the AdaBoost members fit the one staged device
#     copy of X (the exact forest data session DART opens by default);
#   MOJOLEARN_TE_ADA_SESSION_SHARE  with it, later members reuse the first
#     member's quantile tables (may move bits: a quality gate, not a digest).
_TE_NATIVE_SPLITS = 1
_TE_ADA_SESSION = 2
_TE_ADA_SESSION_SHARE = 4


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


def _trees_native_glue(est):
    """`est` when MOJOLEARN_TE_NATIVE_SPLITS selects the device cv
    bookkeeping for it (FAST tier, a binding that carries
    `x_trees_device_folds`), else None."""
    if not _trees_switch(est, _TE_NATIVE_SPLITS):
        return None
    return est if callable(getattr(est._bind(), "x_trees_device_folds", None)) else None


def _trees_ada_session_default(est):
    """The `MOJOLEARN_FOREST_SESSION` default an AdaBoost fit passes to
    `_trees_member_session`: "share" under MOJOLEARN_TE_ADA_SESSION_SHARE,
    "1" under MOJOLEARN_TE_ADA_SESSION, else main's "0" (each member stages
    X itself unless the env says otherwise)."""
    if _trees_switch(est, _TE_ADA_SESSION_SHARE):
        return "share"
    return "1" if _trees_switch(est, _TE_ADA_SESSION) else "0"


def _trees_float_dtype(Y):
    """True for a buffer whose dtype is a float (an ndarray's `kind`, or an
    `Array`'s format string): its `tolist()` labels are Python floats, the
    classes a native float64 column encodes to."""
    d = getattr(Y, "dtype", None)
    kind = getattr(d, "kind", None)
    if kind is not None:
        return kind == "f"
    return isinstance(d, str) and d.lstrip("<>=|").startswith("f")


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
# warm_start is refused by name. oob_score (`_set_oob_score` :1000 / :1310)
# sums each member's output on the rows it never drew, in member order.
class _BaggingBase(_TreesEnsembleBase):
    def __init__(self, estimator, n_estimators, max_samples, max_features, bootstrap,
                 bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose):
        if oob_score and not bootstrap:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
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
    def _batched_ok(self, base, n_rows, n, n_feat, d, sample_weight):
        if os.environ.get("MOJOLEARN_HR2_BAGGING_MEMBERS") == "1":
            return False
        if type(base) not in (DecisionTreeClassifier, DecisionTreeRegressor):
            return False
        if base.splitter != "best" or getattr(base, "max_leaf_nodes", None) is not None:
            return False
        if sample_weight is not None or self.oob_score or self.bootstrap_features or n_feat != d:
            return False
        if not self.bootstrap and n_rows != n:
            return False  # sampling without replacement: not the forest's draw
        return True

    def _fit_batched(self, Xa, y, base, n_rows, n, seed):
        common = dict(n_estimators=int(self.n_estimators), criterion=base.criterion,
                      max_depth=base.max_depth, min_samples_split=base.min_samples_split,
                      min_samples_leaf=base.min_samples_leaf,
                      max_features=1.0 if base.max_features is None else base.max_features,
                      min_impurity_decrease=base.min_impurity_decrease,
                      bootstrap=bool(self.bootstrap),
                      max_samples=(n_rows / n) if self.bootstrap else None,
                      random_state=seed, n_bins=base.n_bins,
                      numeric_mode=getattr(base, "numeric_mode", None))
        if isinstance(base, DecisionTreeClassifier):
            forest = RandomForestClassifier(class_weight=base.class_weight, **common)
        else:
            forest = RandomForestRegressor(**common)
        forest.fit(Xa, y)
        self._batched = (forest, base)
        self._estimators = None
        d = Xa.shape[1]
        self.estimators_features_ = [_trees_arange(d) for _ in range(int(forest._n_trees))]
        self._oob_rows = []
        self.n_features_in_ = d
        return self

    @property
    def estimators_(self):
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
        offsets = forest._offsets.tolist()
        no = int(forest._num_outputs)
        colid, ques = forest._colid.tolist(), forest._quesval.tolist()
        left, leaves = forest._left_child.tolist(), forest._leaves.tolist()
        seed = _trees_seed(self.random_state)
        out = []
        for t in range(int(forest._n_trees)):
            lo, hi = offsets[t], offsets[t + 1]
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, t))
            est._refresh_config()
            est._capture_fit_mode()
            est._offsets = Array.from_list([0, hi - lo], "<i4")
            est._colid = Array.from_list(colid[lo:hi], "<i4")
            est._quesval = Array.from_list(ques[lo:hi], "<f4")
            est._left_child = Array.from_list(left[lo:hi], "<i4")
            est._leaves = Array.from_list(leaves[lo * no:hi * no], "<f4")
            est._n_trees = 1
            est._num_outputs = no
            est.n_features_in_ = forest.n_features_in_
            if hasattr(forest, "classes_"):
                est.classes_ = forest.classes_
                est.n_classes_ = forest.n_classes_
            out.append(est)
        return out

    def _fit_bags(self, X, y_sub, sample_weight, make_default):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        seed = _trees_seed(self.random_state)
        n_rows = self._count(self.max_samples, n, "max_samples")
        n_feat = self._count(self.max_features, d, "max_features")
        base = self.estimator if self.estimator is not None else make_default()
        if self._batched_ok(base, n_rows, n, n_feat, d, sample_weight):
            return self._fit_batched(Xa, y_sub(None, True), base, n_rows, n, seed)
        sw = None if sample_weight is None else as_f32_c(sample_weight, ndim=1, name="sample_weight")[0]
        self.estimators_, self.estimators_features_, self._oob_rows = [], [], []
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
            if self.oob_score:
                self._oob_rows.append(self._oob_of(rows, n, self))
        self.n_features_in_ = d
        return self

    @staticmethod
    def _oob_of(rows, n, est=None):
        """The rows a member never drew, ascending (sklearn `indices_to_mask`
        negated): one native pass (`x_trees_unseen_rows`, on the device in a
        GPU build)."""
        r = rows if isinstance(rows, Array) else Array.from_list(list(rows), "<i4")
        out = empty((max(1, n),), "<i4")
        k = int(_trees_x_bind(est).x_trees_unseen_rows(addr_ro(r, name="rows"), addr(out, name="oob"),
                                                       [len(r), n]))
        return out[:k]

    def _oob_outputs(self, Xa, k, member_out):
        """(sums n x k float64, per-row member count): each member's output on
        its out-of-bag rows, added in member order (sklearn `_set_oob_score`)."""
        n = Xa.shape[0]
        acc = zeros((n * k,), "<f8")
        counts = [0] * n
        for est, cols, oob in zip(self.estimators_, self.estimators_features_, self._oob_rows):
            m = len(oob)
            if m == 0:
                continue
            out = member_out(est, self._gather(Xa, oob, cols), m)
            self._bind().x_trees_accumulate_rows(addr(acc, name="oob"), addr_ro(out, name="member"),
                                                 addr_ro(oob, name="rows"), [n, k, m])
            for i in oob.tolist():
                counts[i] += 1
        return acc, counts

    def _check_X(self, X):
        if getattr(self, "_batched", None) is None and getattr(self, "_estimators", None) is None:
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
        self._fit_bags(X, lambda rows, all_rows: codes if all_rows else self._gather_codes(codes, rows),
                       sample_weight, DecisionTreeClassifier)
        if self.oob_score:
            self._set_oob_score(X, codes)
        return self

    def _member_proba(self, est, Xs, m):
        """A member's class scores on Xs as dense m x k float64 (its predict_proba
        in the columns of the classes it saw, else a one-hot vote)."""
        k = self.n_classes_
        out = zeros((m * k,), "<f8")
        if hasattr(est, "predict_proba"):
            self._acc_cols(out, est.predict_proba(Xs), _trees_sub_cols(est), m, k)
        else:
            self._acc_votes(out, self._codes_of(est.predict(Xs), est.classes_), m, k, 1.0)
        return out

    def _set_oob_score(self, X, codes):
        """sklearn BaggingClassifier._set_oob_score: the out-of-bag class scores
        summed per row, normalised (a row no member left out is uniform 1 / k,
        DEVIATION 5605, where sklearn divides 0 / 0), accuracy of their argmax."""
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, k = Xa.shape[0], self.n_classes_
        acc, _ = self._oob_outputs(Xa, k, self._member_proba)
        self._bind().x_trees_normalize_rows(addr(acc, name="oob"), [n, k])
        self.oob_decision_function_ = acc.reshape((n, k))
        # the per-row match is the native elementwise compare (two int32
        # Arrays), not a Python loop over the rows
        hits = self._argmax(acc, n, k) == as_i32_c(codes, ndim=1, name="codes")[0]
        self.oob_score_ = hits.sum() / n

    def predict_proba(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        if getattr(self, "_batched", None) is not None:
            # lane hr2-gbdt-host: the forest's mean of the trees' leaf
            # distributions, every tree in one device walk
            return self._batched[0].predict_proba(Xa).astype("<f8")
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
        self._fit_bags(X, lambda rows, all_rows: y32 if all_rows else self._gather_vec(y32, rows),
                       sample_weight, DecisionTreeRegressor)
        if self.oob_score:
            self._set_oob_score(X, y32)
        return self

    def _set_oob_score(self, X, y32):
        """sklearn BaggingRegressor._set_oob_score: the out-of-bag predictions
        summed per row over member order, divided by their count (a row no
        member left out divides by 1, as sklearn does), R^2 against y."""
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n = Xa.shape[0]

        def member_out(est, Xs, m):
            p, _ = as_f32_c(est.predict(Xs), ndim=1, name="prediction")
            return Array.from_list(p.tolist(), "<f8")

        acc, counts = self._oob_outputs(Xa, 1, member_out)
        pred = [v / max(c, 1) for v, c in zip(acc.tolist(), counts)]
        self.oob_prediction_ = Array.from_list(pred, "<f8")
        y = y32.tolist()
        mean = math.fsum(y) / n
        ss_tot = math.fsum((v - mean) * (v - mean) for v in y)
        ss_res = math.fsum((a - b) * (a - b) for a, b in zip(y, pred))
        self.oob_score_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)

    def predict(self, X):
        Xa = self._check_X(X)
        n = Xa.shape[0]
        if getattr(self, "_batched", None) is not None:
            # lane hr2-gbdt-host: the forest's mean over the trees
            return as_f32_c(self._batched[0].predict(Xa), ndim=1, name="prediction")[0].astype("<f8")
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
        # A best-splitter DecisionTreeClassifier member fits column-major X
        # with weights: transpose and finite-scan X ONCE for every member
        # (the member's own `fit` did both per member on the same bytes).
        Xcm = None
        member_enc = None
        if type(base) is DecisionTreeClassifier and base.splitter == "best":
            Xcm = _trees_colmajor(Xa, self)
            if not all_finite(Xcm):
                raise ValueError("X contains NaN or infinity; the forest has no missing-value arm")
            # every member encodes the SAME codes: once, for all of them
            # (`MOJOLEARN_ADABOOST_REENCODE=1` re-encodes per member, the
            # A/B arm)
            if os.environ.get("MOJOLEARN_ADABOOST_REENCODE") != "1":
                member_enc = encode_labels(codes)
        # trees-apple3: every member fits the SAME X, staged on the device
        # once for all of them (None: each member stages it, as before)
        session = None
        if Xcm is not None:
            session = _trees_member_session(base, Xcm, False, x_finite=True,
                                            default=_trees_ada_session_default(self))
        try:
            self._fit_members(base, seed, m, Xa, Xcm, codes, w, n, k, member_enc, session, b, stats)
        finally:
            if session is not None:
                session.close()
        self.n_features_in_ = Xa.shape[1]
        return self

    def _fit_members(self, base, seed, m, Xa, Xcm, codes, w, n, k, member_enc, session, b, stats):
        for it in range(m):
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, it))
            if Xcm is not None:
                est._fit_weighted(Xcm, codes, self._w32(w, n), x_finite=True, encoded=member_enc,
                                  session=session)
            else:
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
            d = empty((n,), "<f8")
            self._bind().x_trees_margin2(addr_ro(acc, name="votes"), addr(d, name="margin"), [n, 0])
            return d
        return acc.reshape((n, k))

    def predict_proba(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        acc = self._decision(Xa)
        if k == 2:
            pairs = empty((n * 2,), "<f8")
            self._bind().x_trees_margin2(addr_ro(acc, name="votes"), addr(pairs, name="pairs"), [n, 2])
            acc = pairs
        else:
            self._scale(acc, k - 1)
        self._bind().x_trees_softmax_rows(addr(acc, name="proba"), [n, k])
        return acc.reshape((n, k))

    def predict(self, X):
        Xa = self._check_X(X)
        n, k = Xa.shape[0], self.n_classes_
        acc = self._decision(Xa)
        if k == 2:
            codes = empty((n,), "<i4")
            self._bind().x_trees_margin2(addr_ro(acc, name="votes"), addr(codes, name="codes"), [n, 1])
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
        # trees-apple3: a best-splitter squared-error DecisionTreeRegressor
        # member fits rows of the SAME X, staged on the device once; the
        # member's rows are gathered there (None: gathered on the host and
        # staged per member, as before)
        session = None
        dflt = _trees_ada_session_default(self)
        if (forest_data_session_choice(None, dflt) is not None
                and type(base) is DecisionTreeRegressor and base.splitter == "best"
                and base.criterion in ("squared_error", "mse")
                and hasattr(_trees_member_native(base), "rf_regressor_fit_session_rows_export")):
            session = _trees_member_session(base, Xa, True, default=dflt)
        try:
            self._fit_members(base, seed, m, Xa, y32, w, n, cols, session, b, stats)
        finally:
            if session is not None:
                session.close()
        self.n_features_in_ = d
        return self

    def _fit_members(self, base, seed, m, Xa, y32, w, n, cols, session, b, stats):
        for it in range(m):
            rows = empty((n,), "<i4")
            b.x_trees_weighted_sample(addr_ro(w, name="w"), addr(rows, name="rows"), [n, n, seed, it])
            est = _trees_clone(base, random_state=_trees_sub_seed(seed, it))
            if session is not None:
                est._fit_in_session(session, self._gather_vec(y32, rows), rows=rows)
            else:
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
        for sd in (random_state, drop_seed, feature_fraction_seed, bagging_seed):
            _trees_seed(sd)

    def _tree_nodes(self, tree, Xa):
        n, d = Xa.shape
        out = empty((n,), "<i4")
        self._bind().x_trees_apply(addr_ro(tree._offsets, name="offsets"), addr_ro(tree._colid, name="colid"),
                                   addr_ro(tree._quesval, name="quesval"), addr_ro(tree._left_child, name="left"),
                                   addr_ro(Xa, name="X"), addr(out, name="nodes"), [n, d, 0, 1])
        return out

    def _add(self, score, nodes, values, weight, c=0):
        n = len(nodes)
        self._bind().x_trees_tree_score_add(addr_ro(nodes, name="nodes"), addr_ro(values, name="values"),
                                            addr(score, name="score") + 8 * c * n, [n, float(weight)])

    def _bag(self, n, it):
        """The bagged rows of iteration `it` (row order), or None for all."""
        freq, frac = int(self.subsample_freq), float(self.subsample)
        if freq <= 0 or not frac < 1.0:
            return None
        start = it - it % freq
        if getattr(self, "_bag_at", None) is not None and self._bag_at[0] == start:
            return self._bag_at[1]
        # the rows whose draw is below frac (none: the smallest draw's row),
        # drawn and compacted natively (`x_trees_bag_rows`, on the device in
        # a GPU build); the draws are `x_trees_uniform`'s stream
        out = empty((max(1, n),), "<i4")
        k = int(self._bind().x_trees_bag_rows(addr(out, name="bag"), [n, _trees_seed(self.bagging_seed), start, frac]))
        self._bag_at = (start, out[:k])
        return self._bag_at[1]

    def _cols(self, d, t):
        """The sorted feature subset of tree `t`, or None for all."""
        frac = float(self.colsample_bytree)
        if not frac < 1.0:
            return None
        cnt = max(int(d * frac + 0.5), min(1, d))
        if cnt >= d:
            return None
        drawn = self._indices(d, cnt, False, _trees_seed(self.feature_fraction_seed), t).tolist()
        return Array.from_list(sorted(drawn), "<i4")

    def _boost(self, Xa, y32, n_classes=1):
        n, d = Xa.shape
        K = int(n_classes)
        b = self._bind()
        seed = _trees_seed(self.random_state)
        drop_seed = _trees_seed(self.drop_seed)
        if self._KIND == 0:
            inits = [_trees_fsum_f32(b, y32) / n]
        elif self._KIND == 1:
            p = _trees_fsum_f32(b, y32) / n
            if not 0.0 < p < 1.0:
                raise ValueError("y must hold both classes")
            inits = [float(b.x_trees_log64(p / (1.0 - p)))]
        else:
            counts = [0] * K
            for v in y32.tolist():
                counts[int(v)] += 1
            inits = [float(b.x_trees_log64(max(1e-15, cnt / n))) for cnt in counts]
        self.init_score_ = inits[0] if K == 1 else inits
        score = _class_major_fill(inits, n)
        g, h, target = empty((K * n,), "<f8"), empty((K * n,), "<f8"), empty((K * n,), "<f4")
        lr = float(self.learning_rate)
        l1, mds, lam = float(self.reg_alpha), float(self.max_delta_step), float(self.reg_lambda)
        self.n_classes_ = K
        self.trees_, self.tree_values_, self.tree_coefs_, self.tree_weights_ = [], [], [], []
        self._bag_at = None
        train_nodes = []
        sum_w = 0.0
        max_depth = None if self.max_depth is None or int(self.max_depth) <= 0 else int(self.max_depth)
        all_cols = _trees_arange(d)
        # trees-apple3: members that fit every row and column of X share ONE
        # staged copy of it (a bagged or column-sampled member gathers its
        # own X and fits as before)
        session = None
        if not (float(self.subsample) < 1.0 and int(self.subsample_freq) > 0) \
                and not float(self.colsample_bytree) < 1.0:
            # lane/gap-nv-classical2: DART opens the exact session by default
            # (MOJOLEARN_FOREST_SESSION=0 still turns it off). Without it each
            # of the n_estimators members scanned, transposed, staged and
            # uploaded the whole of X and rebuilt its quantiles; in the
            # session each member still draws its own quantile sample, so
            # each member's forest is the one its own fit returns.
            session = _trees_member_session(
                RandomForestRegressor(n_estimators=1, numeric_mode=self.numeric_mode), Xa, True,
                default="1")
        try:
            if self._dart_device(b, session, K):
                # lane/apple-fast-dart: the round on the device (FAST + Apple
                # binaries (default; not -D MOJOLEARN_DART_DEVICE_OFF) expose the
                # x_trees_dart_* entries; every other binary takes main's loop)
                self._boost_loop_device(Xa, y32, K, b, seed, drop_seed, inits, lr, l1, mds, lam, max_depth,
                                        session)
            else:
                self._boost_loop(Xa, y32, K, b, seed, drop_seed, score, g, h, target, lr, l1, mds, lam,
                                 max_depth, all_cols, session)
        finally:
            if session is not None:
                session.close()
        self._bag_at = None
        self.n_features_in_ = d
        return self

    def _boost_loop(self, Xa, y32, K, b, seed, drop_seed, score, g, h, target, lr, l1, mds, lam,
                    max_depth, all_cols, session):
        n, d = Xa.shape
        train_nodes = []
        sum_w = 0.0
        for it in range(int(self.n_estimators)):
            t = len(self.tree_weights_)
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
                for c in range(K):
                    j = i * K + c
                    self._add(score, train_nodes[j], self.tree_values_[j], -self.tree_coefs_[j], c)
            k = len(drop)
            if not self.xgboost_dart_mode:
                shrink = lr / (1.0 + k)
            else:
                shrink = lr if k == 0 else lr / (lr + k)
            if K == 1:
                b.x_trees_gradients(addr_ro(score, name="score"), addr_ro(y32, name="y"), addr(g, name="g"),
                                    addr(h, name="h"), addr(target, name="target"), [n, self._KIND])
            else:
                b.x_trees_gradients(addr_ro(score, name="score"), addr_ro(y32, name="y"), addr(g, name="g"),
                                    addr(h, name="h"), addr(target, name="target"), [n, 2, K])
            rows = self._bag(n, it)
            for c in range(K):
                j = it * K + c
                cols = self._cols(d, j)
                tgt = target if K == 1 else self._gather_vec(target, Array.from_list(
                    list(range(c * n, (c + 1) * n)), "<i4"))
                Xf, yf = Xa, tgt
                if rows is not None or cols is not None:
                    Xf = self._gather(Xa, rows if rows is not None else _trees_arange(n),
                                      cols if cols is not None else all_cols)
                    if rows is not None:
                        yf = self._gather_vec(tgt, rows)
                tree = RandomForestRegressor(
                    n_estimators=1, bootstrap=False, max_features=1.0, max_depth=max_depth,
                    max_leaves=int(self.num_leaves), min_samples_leaf=int(self.min_child_samples),
                    n_bins=int(self.max_bin), random_state=_trees_sub_seed(seed, j), n_streams=1,
                    numeric_mode=self.numeric_mode)
                if session is not None and Xf is Xa:
                    tree._fit_in_session(session, yf)
                else:
                    tree.fit(Xf, yf)
                if cols is not None:
                    cl = cols.tolist()
                    tree._colid = Array.from_list([cl[v] if v >= 0 else v for v in tree._colid.tolist()], "<i4")
                nodes = self._tree_nodes(tree, Xa)
                n_nodes = int(tree._offsets.tolist()[1])
                values = empty((n_nodes,), "<f4")
                ga, ha = addr_ro(g, name="g") + 8 * c * n, addr_ro(h, name="h") + 8 * c * n
                if rows is None:
                    params = [n, n_nodes, lam] if l1 == 0.0 and mds == 0.0 else [n, n_nodes, lam, l1, mds]
                    b.x_trees_leaf_newton(addr_ro(nodes, name="nodes"), ga, ha, addr(values, name="values"), params)
                else:
                    b.x_trees_leaf_newton_rows(addr_ro(nodes, name="nodes"), addr_ro(rows, name="rows"), ga, ha,
                                               addr(values, name="values"), [n, len(rows), n_nodes, lam, l1, mds])
                self._add(score, nodes, values, shrink, c)
                self.trees_.append(tree)
                self.tree_values_.append(values)
                self.tree_coefs_.append(shrink)
                train_nodes.append(nodes)
            for i in drop:
                if not self.xgboost_dart_mode:
                    factor, wdiv = k / (k + 1.0), 1.0 / (k + 1.0)
                else:
                    factor, wdiv = k / (k + lr), 1.0 / (k + lr)
                for c in range(K):
                    j = i * K + c
                    self.tree_coefs_[j] *= factor
                    self._add(score, train_nodes[j], self.tree_values_[j], self.tree_coefs_[j], c)
                if not self.uniform_drop:
                    sum_w -= self.tree_weights_[i] * wdiv
                    self.tree_weights_[i] *= factor
            self.tree_weights_.append(shrink)
            sum_w += shrink

    # -------------------------------------------- lane/apple-fast-dart
    # The boosting round on the device (xtrees/dart_device.mojo): FAST +
    # Apple only, default unless -D MOJOLEARN_DART_DEVICE_OFF; the only build
    # that registers x_trees_dart_open. Same drop set, shrink factors and
    # tree fits as `_boost_loop`; the score, gradients and leaf values are
    # float32 on the device and the dropped trees come off and go back as
    # one gathered sum per row (the docstring of dart_device.mojo).
    _DART_VALUES_CAP = 1 << 26

    def _dart_device(self, b, session, K):
        if session is None or not callable(getattr(b, "x_trees_dart_open", None)):
            return False
        node_cap = 2 * int(self.num_leaves) - 1
        return 1 <= node_cap <= 65535 and int(self.n_estimators) * K * node_cap <= self._DART_VALUES_CAP

    @staticmethod
    def _dart_thr(v):
        """ceil(v * 2^53) as an int: `u < v` for a counter draw u = m / 2^53
        (m the top 53 bits) is exactly `m < ceil(v * 2^53)`."""
        if not v > 0.0:
            return 0
        if v >= 1.0:
            return 1 << 53
        return int(math.ceil(v * 9007199254740992.0))

    def _boost_loop_device(self, Xa, y32, K, b, seed, drop_seed, inits, lr, l1, mds, lam, max_depth, session):
        n, d = Xa.shape
        n_iters = int(self.n_estimators)
        node_cap = 2 * int(self.num_leaves) - 1
        inits32 = Array.from_list([float(v) for v in inits], "<f4")
        skip_thr = self._dart_thr(float(self.skip_drop))
        bad = empty((1,), "<i4")
        handle = b.x_trees_dart_open(addr_ro(Xa, name="X"), addr_ro(y32, name="y"), addr_ro(inits32, name="inits"),
                                     [n, d, K, self._KIND, n_iters, node_cap])
        try:
            sum_w = 0.0
            for it in range(n_iters):
                t = len(self.tree_weights_)
                thr = [0] * t
                if t:
                    rate = float(self.drop_rate)
                    if not self.uniform_drop:
                        inv_avg = t / sum_w if sum_w > 0 else 0.0
                        if int(self.max_drop) > 0 and sum_w > 0:
                            rate = min(rate, int(self.max_drop) * inv_avg / sum_w)
                        thr = [self._dart_thr(rate * self.tree_weights_[i] * inv_avg) for i in range(t)]
                    else:
                        if int(self.max_drop) > 0:
                            rate = min(rate, int(self.max_drop) / t)
                        thr = [self._dart_thr(rate)] * t
                coef32 = Array.from_list([float(v) for v in self.tree_coefs_] or [0.0], "<f4")
                thr64 = Array.from_list(thr or [0], "<i8")
                flags = empty((max(t, 1),), "<i4")
                targets = [empty((n,), "<f4") for _ in range(K)]
                b.x_trees_dart_step(handle, addr_ro(coef32, name="coef"), addr_ro(thr64, name="thr"),
                                    addr(flags, name="flags"), addr(bad, name="bad"),
                                    [addr(tg, name="target") for tg in targets], [t, drop_seed, it, skip_thr])
                if bad.tolist()[0]:
                    raise RuntimeError("x_trees dart: a tree walk left its tree (child or column out of range)")
                fl = flags.tolist()
                drop = [i for i in range(t) if fl[i]]
                k = len(drop)
                if not self.xgboost_dart_mode:
                    shrink = lr / (1.0 + k)
                    factor = k / (k + 1.0)
                    wdiv = 1.0 / (k + 1.0)
                else:
                    shrink = lr if k == 0 else lr / (lr + k)
                    factor = k / (k + lr)
                    wdiv = 1.0 / (k + lr)
                for c in range(K):
                    j = it * K + c
                    tree = RandomForestRegressor(
                        n_estimators=1, bootstrap=False, max_features=1.0, max_depth=max_depth,
                        max_leaves=int(self.num_leaves), min_samples_leaf=int(self.min_child_samples),
                        n_bins=int(self.max_bin), random_state=_trees_sub_seed(seed, j), n_streams=1,
                        numeric_mode=self.numeric_mode)
                    tree._fit_in_session(session, targets[c])
                    offs = tree._offsets.tolist()
                    lo, n_nodes = int(offs[0]), int(offs[1]) - int(offs[0])
                    values = empty((n_nodes,), "<f4")
                    b.x_trees_dart_add(handle, addr_ro(tree._colid, name="colid"),
                                       addr_ro(tree._quesval, name="quesval"), addr_ro(tree._left_child, name="left"),
                                       addr(values, name="values"),
                                       [j, c, lo, n_nodes, float(shrink), float(factor), lam, l1, mds])
                    self.trees_.append(tree)
                    self.tree_values_.append(values)
                    self.tree_coefs_.append(shrink)
                for i in drop:
                    for c in range(K):
                        self.tree_coefs_[i * K + c] *= factor
                    if not self.uniform_drop:
                        sum_w -= self.tree_weights_[i] * wdiv
                        self.tree_weights_[i] *= factor
                self.tree_weights_.append(shrink)
                sum_w += shrink
        finally:
            b.x_trees_dart_close(handle, addr(bad, name="bad"))
        if bad.tolist()[0]:
            raise RuntimeError("x_trees dart: a tree walk left its tree (child or column out of range)")

    def _raw(self, X):
        """Class-major raw scores (K * n) float64; K = 1 is (n,)."""
        if not hasattr(self, "trees_"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        if Xa.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xa.shape[1]} features, fit saw {self.n_features_in_}")
        n, K = Xa.shape[0], int(getattr(self, "n_classes_", 1))
        inits = [self.init_score_] if K == 1 else list(self.init_score_)
        score = _class_major_fill(inits, n)
        for j, (tree, values, coef) in enumerate(zip(self.trees_, self.tree_values_, self.tree_coefs_)):
            self._add(score, self._tree_nodes(tree, Xa), values, coef, j % K)
        return score

    def _raw_rows(self, X):
        """The raw scores as (n, K) row-major, K >= 2."""
        raw = self._raw(X)
        K = self.n_classes_
        n = len(raw) // K
        out = empty((n * K,), "<f8")
        self._bind().x_trees_transpose_f64(addr_ro(raw, name="raw"), addr(out, name="raw rows"), [K, n])
        return out.reshape((n, K))


class DARTRegressor(_DARTBase):
    """LightGBM's `boosting='dart'` with the L2 objective. `predict` is the
    raw score, float64."""
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
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        y32, _ = as_f32_c(y, ndim=1, name="y")
        if len(y32) != Xa.shape[0]:
            raise ValueError(f"y has {len(y32)} rows, X has {Xa.shape[0]}")
        return self._boost(Xa, y32)

    def predict(self, X):
        return self._raw(X)


class DARTClassifier(_DARTBase):
    """LightGBM's `boosting='dart'`: binary logloss for two classes
    (`predict_proba` is [1 - p, p], p the sigmoid of the raw score, a
    two-column softmax of [0, raw]), multiclass softmax for more (one tree
    per class per iteration; `predict_proba` the softmax of the K raw
    scores)."""
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
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        self.classes_, codes = encode_labels(y)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("y must hold at least two classes")
        if len(codes) != Xa.shape[0]:
            raise ValueError(f"y has {len(codes)} rows, X has {Xa.shape[0]}")
        y32, _ = as_f32_c(codes, ndim=1, name="y")
        self._KIND = 1 if K == 2 else 2
        return self._boost(Xa, y32, 1 if K == 2 else K)

    def decision_function(self, X):
        if int(getattr(self, "n_classes_", 1)) > 1:
            return self._raw_rows(X)
        return self._raw(X)

    def predict_proba(self, X):
        if int(getattr(self, "n_classes_", 1)) > 1:
            acc = self._raw_rows(X)
            n, K = acc.shape
            acc = acc.reshape((n * K,))
            self._bind().x_trees_softmax_rows(addr(acc, name="proba"), [n, K])
            return acc.reshape((n, K))
        raw = self._raw(X).tolist()
        acc = Array.from_list([v for r in raw for v in (0.0, r)], "<f8")
        self._bind().x_trees_softmax_rows(addr(acc, name="proba"), [len(raw), 2])
        return acc.reshape((len(raw), 2))

    def predict(self, X):
        if int(getattr(self, "n_classes_", 1)) > 1:
            acc = self._raw_rows(X)
            n, K = acc.shape
            return decode_labels(self.classes_, self._argmax(acc.reshape((n * K,)), n, K))
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
    """An int number of folds (None: `default`), or `cv` itself when it is a
    splitter (has `split`) or an iterable of (train, test) index pairs."""
    if cv is None:
        return default
    if hasattr(cv, "split") or (not isinstance(cv, (str, bytes, numbers.Number)) and hasattr(cv, "__iter__")):
        return cv
    if is_bool(cv) or not isinstance(cv, numbers.Integral) or cv < 2:
        raise ValueError(f"cv={cv!r}: an int >= 2, a splitter with split(X, y), or an iterable of"
                         " (train, test) index pairs")
    return int(cv)


def _trees_fold_rows(folds, i):
    tr = [r for r, f in enumerate(folds) if f != i]
    te = [r for r, f in enumerate(folds) if f == i]
    return Array.from_list(tr, "<i4"), Array.from_list(te, "<i4")


class _FoldRows(list):
    """The [(train rows, test rows)] of `_trees_native_folds`: int32 views into
    the one buffer the device filled, kept alive here."""
    __slots__ = ("_owner",)


def _trees_native_folds(est, n_splits, n, codes, n_classes=0):
    """`_trees_splits` for an int cv with the fold assignment and the fold
    row lists made on the device (xtrees/folds_device.mojo through
    `x_trees_device_folds`: sklearn's unshuffled StratifiedKFold / KFold law,
    the same folds `_trees_stratified_folds` / `_trees_kfolds` build, no
    Python row loop and no host row loop), under `-D MOJOLEARN_TE_NATIVE_SPLITS`
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
    cnt = [int(v) for v in counts.tolist()[:n_splits]]
    used = max([i for i in range(n_splits) if cnt[i] > 0] or [0]) + 1
    out = _FoldRows()
    out._owner = rows
    for i in range(used):
        base = rows._addr + 4 * i * n
        tr = memory_at(base, 4 * (n - cnt[i]), writable=False).cast("i")
        te = memory_at(base + 4 * (n - cnt[i]), 4 * cnt[i], writable=False).cast("i")
        out.append((tr, te))
    return out


def _trees_splits(cv, X, y, n, codes=None, partition=False, native=None, n_classes=0):
    """[(train rows, test rows)] as int32 Arrays. An int is sklearn's
    unshuffled StratifiedKFold (with `codes`) or KFold; a splitter object's
    `split(X, y)` and an iterable of pairs are taken as given (sklearn
    `check_cv`), their indices in the order they come. `partition`: every row
    must be in exactly one test set (sklearn cross_val_predict). `native`:
    the estimator whose binding builds an int cv's folds on the device
    (`_trees_native_folds`; `codes` is then an int32 Array in
    [0, n_classes)), else None."""
    c = _trees_cv(cv)
    if isinstance(c, int):
        if native is not None:
            return _trees_native_folds(native, c, n, codes, n_classes)
        folds = _trees_stratified_folds(codes, c) if codes is not None else _trees_kfolds(n, c)
        return [_trees_fold_rows(folds, i) for i in range(max(folds) + 1)]
    pairs = c.split(X, y) if hasattr(c, "split") else c
    out, seen = [], [0] * n
    for pair in pairs:
        tr, te = pair
        tr = [int(v) for v in (tr.tolist() if hasattr(tr, "tolist") else tr)]
        te = [int(v) for v in (te.tolist() if hasattr(te, "tolist") else te)]
        for v in tr + te:
            if not 0 <= v < n:
                raise ValueError(f"cv index {v} is outside [0, {n})")
        if not tr or not te:
            raise ValueError("every cv split needs train and test rows")
        for v in te:
            seen[v] += 1
        out.append((Array.from_list(tr, "<i4"), Array.from_list(te, "<i4")))
    if not out:
        raise ValueError("cv produced no splits")
    if partition and any(v != 1 for v in seen):
        raise ValueError("cross_val_predict only works for partitions: every row must be in exactly one test set")
    return out


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
# folds (StratifiedKFold for the classifier, KFold for the regressor), a
# splitter object or an iterable of (train, test) pairs whose test sets
# partition the rows (`_trees_splits`).
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

    def _fit_stack(self, X, y_fit, splits, final):
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
        cols = _trees_arange(d)
        for tr, te in splits:
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
        nat = _trees_native_glue(self)
        splits = _trees_splits(self.cv, X, y, len(codes), codes=codes if nat is not None else codes.tolist(),
                               partition=True, native=nat, n_classes=len(self.classes_))
        final = self.final_estimator
        if final is None:
            from .linear_model import LogisticRegression
            final = LogisticRegression()
        return self._fit_stack(X, lambda rows: codes if rows is None else self._gather_codes(codes, rows),
                               splits, final)

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
        splits = _trees_splits(self.cv, X, y, len(y32), partition=True, native=_trees_native_glue(self))
        final = self.final_estimator
        if final is None:
            from .linear_model import Ridge
            final = Ridge()
        return self._fit_stack(X, lambda rows: y32 if rows is None else self._gather_vec(y32, rows), splits, final)

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
        Ya = None
        if _trees_native_glue(self) is not None and _trees_float_dtype(Y):
            # a float Y: each column natively (x_trees_column_f64), the same
            # float labels its `tolist()` rows hold
            Ya, _ = as_f64_c(Y, ndim=2, name="Y")
            rows = None
            if Ya.shape[0] != Xa.shape[0]:
                raise ValueError(f"Y has {Ya.shape[0]} rows, X has {Xa.shape[0]}")
            m = Ya.shape[1]
        else:
            rows = Y.tolist() if hasattr(Y, "tolist") else [list(r) for r in Y]
            if len(rows) != Xa.shape[0]:
                raise ValueError(f"Y has {len(rows)} rows, X has {Xa.shape[0]}")
            m = len(rows[0])
        self.estimators_, self.classes_ = [], []
        for j in range(m):
            if Ya is not None:
                n = Ya.shape[0]
                col = empty((n,), "<f8")
                self._bind().x_trees_column_f64(addr_ro(Ya, name="Y"), addr(col, name="column"), [n, m, j])
                classes, codes = encode_labels(col)
            else:
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
        if _trees_native_glue(self) is not None:
            # the 0/1 targets natively (x_trees_indicator_codes), the int32
            # words the list comprehension below builds
            targets = [codes] if k == 2 else [self._indicator(codes, j) for j in range(k)]
        else:
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

    def _indicator(self, codes, j):
        n = len(codes)
        yj = empty((n,), "<i4")
        self._bind().x_trees_indicator_codes(addr_ro(codes, name="codes"), addr(yj, name="y"), 0, [n, j])
        return yj

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
#: lane py-misc-prep: CalibratedClassifierCV's per-class epilogue in
#: xtrees (strided calibrators, native columns and 0/1 targets); False (or
#: MOJOLEARN_HOTPATH=python) is the Python reference route (the before arm).
_CAL_NATIVE = True


def _cal_native(est):
    from ._buffer import hotpath_enabled
    return _CAL_NATIVE and hotpath_enabled() and hasattr(est._bind(), "x_trees_platt_apply_strided")


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
        if _cal_native(self):
            return self._fit_calibrators_native(S, codes)
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

    def _fit_calibrators_native(self, S, codes):
        """`_fit_calibrators` with the class column and its 0/1 target made
        natively (x_trees_column_f64, x_trees_indicator_codes): the same
        float64 and int32 words the list comprehensions built."""
        n, c = S.shape
        b = self._bind()
        codes = as_i32_c(codes, ndim=1, name="codes")[0]
        Sa = as_f64_c(S, ndim=2, name="scores")[0] if c > 1 else None
        yj = empty((n,), "<i4")
        y64 = empty((n,), "<f8") if self.method != "sigmoid" else None
        cals = []
        for j in range(c):
            cls = 1 if c == 1 else j
            if c == 1:
                f = S.reshape((n,))
            else:
                f = empty((n,), "<f8")
                b.x_trees_column_f64(addr_ro(Sa, name="S"), addr(f, name="f"), [n, c, j])
            b.x_trees_indicator_codes(addr_ro(codes, name="codes"), addr(yj, name="y"),
                                      addr(y64, name="y64") if y64 is not None else 0, [n, cls])
            if self.method == "sigmoid":
                ab = zeros((2,), "<f8")
                b.x_trees_platt_fit(addr_ro(f, name="f"), addr_ro(yj, name="y"), addr(ab, name="ab"), [n])
                cals.append(("sigmoid", tuple(ab.tolist())))
            else:
                kx, ky = empty((n,), "<f8"), empty((n,), "<f8")
                m = int(b.x_trees_isotonic_fit(addr_ro(f, name="x"), addr_ro(y64, name="y"), addr(kx, name="kx"),
                                               addr(ky, name="ky"), [n]))
                cals.append(("isotonic", (kx[0:m], ky[0:m], m)))
        return cals

    def _calibrated_native(self, e, cals, Xa):
        """`_calibrated` with each calibrator reading its column of the score
        block in place and writing its column of the (n, k) block (binary:
        column 1, then column 0 = 1 - column 1, one IEEE subtract), then the
        same x_trees_normalize_rows: no per-class list and no Python
        interleave. The per-element operations are the contiguous entries'."""
        n, k = Xa.shape[0], len(self.classes_)
        S = as_f64_c(self._scores(e, Xa), ndim=2, name="scores")[0]
        c = S.shape[1]
        b = self._bind()
        w = 2 if k == 2 else k
        acc = empty((n * w,), "<f8")
        for j, (kind, par) in enumerate(cals):
            col = 1 if k == 2 else j
            if kind == "sigmoid":
                b.x_trees_platt_apply_strided(addr_ro(S, name="f"), addr(acc, name="p"),
                                              [n, par[0], par[1], n * c, c, j, n * w, w, col])
            else:
                kx, ky, m = par
                b.x_trees_isotonic_predict_strided(addr_ro(kx, name="kx"), addr_ro(ky, name="ky"),
                                                   addr_ro(S, name="t"), addr(acc, name="p"),
                                                   [m, n, n * c, c, j, n * w, w, col])
        if k == 2:
            b.x_trees_complement_pairs(addr(acc, name="proba"), [n])
            return acc
        b.x_trees_normalize_rows(addr(acc, name="proba"), [n, k])
        return acc

    def _calibrated(self, e, cals, Xa):
        if _cal_native(self):
            return self._calibrated_native(e, cals, Xa)
        n, k = Xa.shape[0], len(self.classes_)
        S = self._scores(e, Xa)
        b = self._bind()
        # class-major (k, n): calibrator j writes row j in place (binary:
        # the one calibrator writes row 1, the positive class)
        cm = empty((k * n,), "<f8")
        for j, (kind, par) in enumerate(cals):
            f = self._column64(S, j)
            dst = addr(cm, name="p") + 8 * (j + (1 if k == 2 else 0)) * n
            if kind == "sigmoid":
                b.x_trees_platt_apply(addr_ro(f, name="f"), dst, [n, par[0], par[1]])
            else:
                kx, ky, m = par
                b.x_trees_isotonic_predict(addr_ro(kx, name="kx"), addr_ro(ky, name="ky"), addr_ro(f, name="t"),
                                           dst, [m, n])
        acc = empty((n * k,), "<f8")
        b.x_trees_transpose_f64(addr_ro(cm, name="p"), addr(acc, name="proba"), [k, n])
        if k == 2:
            # (1 - p, p) rows: the complement into the even cells
            b.x_trees_complement_pairs(addr(acc, name="proba"), [n])
            return acc
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
        nat = _trees_native_glue(self)
        splits = _trees_splits(self.cv, X, y, n, codes=codes if nat is not None else codes.tolist(),
                               partition=not self.ensemble, native=nat, n_classes=len(self.classes_))
        cols = _trees_arange(d)
        self.calibrated_classifiers_ = []
        if self.ensemble:
            for tr, te in splits:
                e = _trees_clone(base)
                e.fit(self._gather(Xa, tr, cols), self._gather_codes(codes, tr))
                Xte = self._gather(Xa, te, cols)
                cals = self._fit_calibrators(self._scores(e, Xte), self._gather_codes(codes, te))
                self.calibrated_classifiers_.append((e, cals))
        else:
            k = len(self.classes_)
            c = 1 if k == 2 else k
            S = zeros((n * c,), "<f8")
            for tr, te in splits:
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
# xtrees/shap.mojo (float32, per-leaf paths, fixed folds) over this
# library's flat forests (RandomForest*, ExtraTrees*, DecisionTree*) and
# DART, on the device on GPU installs. DEVIATION: the node cover is the
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
    `shap_values(X)` is (n, d) for one output, else (n, d, k), float32;
    `expected_value` is a float or a float32 Array of k.

    GPU installs run it on the device (xtrees/shap_device.mojo), CPU-only
    installs on the host (xtrees/shap_host.mojo); both run the units of
    xtrees/shap.mojo in the same fixed orders, so the values are the same
    bits on every column. Float32 throughout (Metal has no float64)."""

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
            init = float(model.init_score_) if K == 1 else [float(v) for v in model.init_score_]
            # one forest of every boosted tree, in boosting order, each tree
            # scaled by its coefficient; a multiclass tree scores one class:
            # its leaf values in column j % K.
            offs, col, q, lc, lv, ts = [0], [], [], [], [], []
            for j, (tree, values, coef) in enumerate(zip(model.trees_, model.tree_values_, model.tree_coefs_)):
                vals = values.tolist()
                if K > 1:
                    c = j % K
                    vals = [v if qq == c else 0.0 for v in vals for qq in range(K)]
                to = tree._offsets.tolist()
                if len(to) != 2:
                    raise ValueError("a DART tree must hold one tree")
                col += tree._colid.tolist()
                q += tree._quesval.tolist()
                lc += tree._left_child.tolist()
                lv += vals
                offs.append(offs[-1] + to[1] - to[0])
                ts.append(float(coef))
            arrays = (Array.from_list(offs, "<i4"), Array.from_list(col, "<i4"), Array.from_list(q, "<f4"),
                      Array.from_list(lc, "<i4"), Array.from_list(lv, "<f4"))
            tscale = Array.from_list(ts, "<f4")
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
        self._bind().x_trees_tree_shap_prepare([addr_ro(a, name="forest") for a in arrays],
                                               addr_ro(tscale, name="tscale"), addr_ro(bg, name="data"),
                                               addr(cover, name="cover"), addr(ev, name="ev"), addr(meta, name="meta"),
                                               [bg.shape[0], bg.shape[1], n_trees, k, n_nodes])
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

    def shap_values(self, X):
        Xa, _ = as_f32_c(X, ndim=2, name="X")
        n, d = Xa.shape
        if d != self.n_features_in_:
            raise ValueError(f"X has {d} features, data has {self.n_features_in_}")
        n_trees, k, n_nodes, slots, width = self._shape
        phi = zeros((n * d * k,), "<f4")
        self._bind().x_trees_tree_shap([addr_ro(a, name="forest") for a in self._forest],
                                       addr_ro(self._tscale, name="tscale"), addr_ro(self._cover, name="cover"),
                                       addr_ro(Xa, name="X"), addr(phi, name="phi"),
                                       [n, d, n_trees, k, n_nodes, slots, width])
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

    def _perm_synthetic(self, x_row, inv, n_perm, syn):
        """Every permutation coalition over every background row, written on
        the device (`xtrees/perm_device.mojo`) into `syn` (allocated on the
        first row, reused after)."""
        bg = self._bg
        nb, d = bg.shape
        total = n_perm * (2 * d + 1) * nb * d
        if syn is None or syn.shape[0] != total:
            syn = empty((total,), "<f4")
        inv_a = Array.from_list(inv, "<i4")
        self._bind().x_trees_perm_synthetic(addr_ro(x_row, name="x"), addr_ro(bg, name="data"),
                                            addr_ro(inv_a, name="inv"), addr(syn, name="synthetic"),
                                            [nb, d, n_perm])
        return syn

    def _mean_over_background(self, syn, m):
        """`_coalitions`'s model call and background mean over a written
        synthetic matrix of m coalitions."""
        nb, d = self._bg.shape
        out = self._eval(syn.reshape((m * nb, d)))
        k = out.shape[1]
        ey = empty((m * k,), "<f8")
        self._bind().x_trees_block_mean(addr_ro(out, name="y"), addr(ey, name="ey"), [m, nb, k])
        return ey

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

    @staticmethod
    def _binom(n, r):
        from ._portable_math import comb
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
            self._link(fx)
            phi = zeros((d * k,), "<f8")
            if M == 1 or ns < 1:
                phi = Array.from_list([fx.tolist()[j] - self._fnull.tolist()[j] for j in range(k)], "<f8")
            else:
                mlist, wlist = self._masks(M, ns, seed, i)
                masks, ey = self._coalitions(x_row, mlist)
                self._link(ey)
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
        syn = None
        for i in range(n):
            x_row = self._gather(Xa, Array.from_list([i], "<i4"), cols).reshape((d,))
            u = empty((max(1, npermutations * d),), "<f8")
            self._bind().x_trees_uniform(addr(u, name="u"), [npermutations * d, seed, i])
            uv = u.tolist()
            perms, inv = [], []
            for p in range(int(npermutations)):
                perm = list(range(d))
                for a in range(d - 1, 0, -1):
                    j = int(uv[p * d + a] * (a + 1))
                    perm[a], perm[j] = perm[j], perm[a]
                perms.append(perm)
                iv = [0] * d
                for pos, f in enumerate(perm):
                    iv[f] = pos
                inv.extend(iv)
            # lane/gap-nv-classical2: the coalitions (all features off, the
            # forward walk adding perm's features, the backward walk removing
            # them) are written on the device from the inverses, into one
            # buffer reused across rows; the same words as the mask lists
            # through `mask_expand`.
            syn = self._perm_synthetic(x_row, inv, int(npermutations), syn)
            ey = self._mean_over_background(syn, int(npermutations) * (2 * d + 1))
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
