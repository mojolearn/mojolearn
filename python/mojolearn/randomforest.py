"""Random Forest on the GPU: cuML's forest, cuML's defaults.

The learner is `ensemble/`: the port of cuML's `ML::RandomForest`
(`randomforest.cuh`) with its batched-levelalgo tree builder, quantile
binning, and the with-replacement `RowSampler` -- THIS wrapper is that
sampler's first Python caller (the `extratrees` surface refuses
`bootstrap=True` by name because its copy has no caller).

THE DEFAULTS ARE cuML's, NOT scikit-learn's, per the package rule
("the defaults follow the upstream each algorithm mirrors"). The two that
will surprise an sklearn user, documented on the classes as well:

* `max_depth` defaults to 16 (cuML `randomforestclassifier.pyx`), not
  sklearn's None/unbounded.
* splits are searched over at most `n_bins` (default 128) per-feature
  QUANTILES, cuML's design, not sklearn's exact thresholds. Faster, and a
  different algorithm -- a comparison against sklearn must say so.

EVERY sklearn-SHAPED PARAMETER IS EITHER HONOURED OR REFUSED BY NAME. None
is accepted and ignored.

`random_state=None` is DETERMINISTIC here (seed 0), same deviation as the
extratrees surface and for the same reason: this library's claim is
bit-reproducibility, so a nondeterministic default entry point would be the
wrong default. NOTE cuML's own caveat, measured by the ensemble lane: the
forest is reproducible from its seed ON ONE GPU MODEL; cuML itself does not
reproduce across GPU models below `n_sampled_rows > 4 * SM * 256`.

X IS COPIED TWICE PER FIT (NumPy to column-major float32, then across the
boundary); on large matrices that is the dominant cost of the CALL.
"""

import numpy as np

from . import _mojolearn_rf
from ._arrays import _addr, _addr_ro, as_f32_c

_CUML_DEFAULT_MAX_DEPTH = 16
_CUML_DEFAULT_N_BINS = 128
_CUML_DEFAULT_N_STREAMS = 4
_CUML_DEFAULT_MAX_BATCH = 4096


def _refuse(name, why):
    raise NotImplementedError(
        f"{name} is not ported: {why} Refused by name rather than accepted"
        " and ignored."
    )


def _max_features_fraction(max_features, n_features):
    """sklearn's max_features forms into the fraction the binding takes,
    resolved the way cuML's python layer resolves them."""
    if max_features is None:
        return 1.0
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return float(np.sqrt(n_features)) / n_features
        if max_features == "log2":
            return float(np.log2(max(2, n_features))) / n_features
        raise ValueError(
            f"max_features={max_features!r} is not a recognised form;"
            " accepted are 'sqrt', 'log2', None, a float fraction or an"
            " int count"
        )
    if isinstance(max_features, (int, np.integer)) and not isinstance(
        max_features, bool
    ):
        if not 1 <= int(max_features) <= n_features:
            raise ValueError(
                f"max_features={max_features} is outside [1, {n_features}]"
            )
        return int(max_features) / n_features
    f = float(max_features)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"max_features={f} is outside (0, 1]")
    return f


class _RandomForestBase:
    def __init__(
        self,
        n_estimators,
        max_depth,
        max_leaf_nodes,
        max_features,
        n_bins,
        min_samples_leaf,
        min_samples_split,
        min_impurity_decrease,
        bootstrap,
        max_samples,
        random_state,
        n_streams,
        max_batch_size,
        oob_score,
        warm_start,
        ccp_alpha,
        class_weight,
        monotonic_cst,
        n_jobs,
        verbose,
        device,
    ):
        if device != "gpu":
            raise ValueError(
                f"device must be 'gpu', got {device!r}: ensemble/ has no"
                " host transcription of the forest builder"
            )
        if oob_score:
            _refuse("oob_score=True", "the engine computes it"
                    " (`fit_forest(oob_score=True)`) but this boundary does"
                    " not carry it yet.")
        if warm_start:
            _refuse("warm_start=True", "there is no incremental fit here,"
                    " and accepting it would silently refit from scratch.")
        if ccp_alpha:
            _refuse("ccp_alpha", "cost-complexity pruning is a"
                    " post-processing pass neither cuML nor this port"
                    " implements. Only 0.0 is accepted.")
        if class_weight is not None:
            _refuse("class_weight", "the engine's apply_class_weight exists"
                    " but this boundary does not carry weights yet.")
        if monotonic_cst is not None:
            _refuse("monotonic_cst", "no cuML counterpart.")
        if n_jobs is not None:
            _refuse("n_jobs", "there is no CPU thread pool here -- the fit"
                    " is one host thread driving the GPU.")
        if verbose:
            _refuse("verbose", "nothing in the Mojo layer logs per-tree"
                    " progress.")
        if max_samples is not None and not 0.0 < float(max_samples) <= 1.0:
            raise ValueError(
                f"max_samples={max_samples} is outside (0, 1]"
            )
        self.device = device
        self._cfg = dict(
            n_estimators=int(n_estimators),
            max_depth=(
                _CUML_DEFAULT_MAX_DEPTH if max_depth is None
                else int(max_depth)
            ),
            max_leaves=(-1 if max_leaf_nodes is None else int(max_leaf_nodes)),
            max_features=max_features,
            n_bins=int(n_bins),
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=int(min_samples_split),
            min_impurity_decrease=float(min_impurity_decrease),
            bootstrap=bool(bootstrap),
            max_samples=(1.0 if max_samples is None else float(max_samples)),
            random_state=(0 if random_state is None else int(random_state)),
            n_streams=int(n_streams),
            max_batch_size=int(max_batch_size),
        )

    def _fit_params(self, n_rows, n_features, n_classes):
        """The 16-slot params list, in _mojolearn_rf.mojo's exact order:
        n_rows, n_cols, n_classes, n_trees, max_depth, max_leaves,
        max_features_fraction, max_n_bins, min_samples_leaf,
        min_samples_split, min_impurity_decrease, bootstrap, max_samples,
        seed, n_streams, max_batch_size."""
        cfg = self._cfg
        return [
            int(n_rows),
            int(n_features),
            int(n_classes),
            cfg["n_estimators"],
            cfg["max_depth"],
            cfg["max_leaves"],
            _max_features_fraction(cfg["max_features"], n_features),
            cfg["n_bins"],
            cfg["min_samples_leaf"],
            cfg["min_samples_split"],
            cfg["min_impurity_decrease"],
            1 if cfg["bootstrap"] else 0,
            cfg["max_samples"],
            cfg["random_state"],
            cfg["n_streams"],
            cfg["max_batch_size"],
        ]

    def _fit_arrays(self, X, y_arr, n_classes, fit_fn):
        Xa = np.asarray(X)
        if Xa.ndim != 2:
            raise ValueError(
                f"X must be 2-D, got {Xa.ndim}-D shape {Xa.shape}"
            )
        n_rows, n_features = Xa.shape
        if len(y_arr) != n_rows:
            raise ValueError(f"y has {len(y_arr)} rows, X has {n_rows}")
        # Column-major is the builder's layout (cuML's `data` is
        # column-major); asfortranarray is that copy, named in the module
        # docstring.
        Xf = np.asfortranarray(Xa, dtype=np.float32)
        params = self._fit_params(n_rows, n_features, n_classes)
        out = fit_fn(_addr_ro(Xf), _addr_ro(y_arr), params)
        del Xf  # the borrow ends with the call
        offsets, colid, quesval, left_child, leaves, meta = out
        self._offsets = np.asarray(offsets, dtype=np.int32)
        self._colid = np.asarray(colid, dtype=np.int32)
        self._quesval = np.asarray(quesval, dtype=np.float32)
        self._left_child = np.asarray(left_child, dtype=np.int32)
        self._leaves = np.asarray(leaves, dtype=np.float32)
        self.n_features_in_ = int(n_features)
        self._n_trees = int(meta[0])
        self._num_outputs = int(meta[1])
        return self

    def _check_predict_input(self, X):
        if not hasattr(self, "_offsets"):
            raise RuntimeError("this estimator is not fitted yet")
        Xa, _ = as_f32_c(X, "X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, fit saw {self.n_features_in_}"
            )
        return Xa, n_rows, n_features


class RandomForestClassifier(_RandomForestBase):
    """cuML's `RandomForestClassifier`, honoured or refused by name.

    Gini splits over per-feature quantiles (`n_bins`, default 128);
    `max_depth` defaults to cuML's 16, not sklearn's None. `predict` is the
    argmax of `predict_proba`, exactly as `RandomForest::predict` argmaxes
    (`randomforest.cuh:417-427`).
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
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        n_bins=_CUML_DEFAULT_N_BINS,
        n_streams=_CUML_DEFAULT_N_STREAMS,
        max_batch_size=_CUML_DEFAULT_MAX_BATCH,
        device="gpu",
    ):
        if criterion != "gini":
            _refuse(
                f"criterion={criterion!r}",
                "only 'gini' crosses this boundary; the engine also"
                " transcribes entropy but the binding does not carry the"
                " selector yet.",
            )
        if min_weight_fraction_leaf:
            _refuse("min_weight_fraction_leaf",
                    "sample weights do not cross this boundary yet.")
        super().__init__(
            n_estimators, max_depth, max_leaf_nodes, max_features, n_bins,
            min_samples_leaf, min_samples_split, min_impurity_decrease,
            bootstrap, max_samples, random_state, n_streams, max_batch_size,
            oob_score, warm_start, ccp_alpha, class_weight, monotonic_cst,
            n_jobs, verbose, device,
        )

    def fit(self, X, y):
        ya = np.asarray(y).ravel()
        self.classes_, codes = np.unique(ya, return_inverse=True)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("y has fewer than 2 classes")
        y32 = np.ascontiguousarray(codes, dtype=np.int32)
        return self._fit_arrays(
            X, y32, self.n_classes_, _mojolearn_rf.rf_classifier_fit
        )

    def predict_proba(self, X):
        Xa, n_rows, n_features = self._check_predict_input(X)
        out = np.empty(n_rows * self._num_outputs, dtype=np.float32)
        wrote = _mojolearn_rf.rf_predict_proba(
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
                f"rf_predict_proba wrote {wrote} of {n_rows} rows"
            )
        return out.reshape(n_rows, self._num_outputs)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_RandomForestBase):
    """cuML's `RandomForestRegressor` (MSE criterion), honoured or refused
    by name. Same quantile-split and `max_depth=16` defaults as the
    classifier; `max_features` defaults to 1.0, cuML's regressor default.
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
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        n_bins=_CUML_DEFAULT_N_BINS,
        n_streams=_CUML_DEFAULT_N_STREAMS,
        max_batch_size=_CUML_DEFAULT_MAX_BATCH,
        device="gpu",
    ):
        if criterion != "squared_error":
            _refuse(
                f"criterion={criterion!r}",
                "only 'squared_error' (cuML's MSE) crosses this boundary;"
                " the engine transcribes more but the binding does not"
                " carry the selector yet.",
            )
        if min_weight_fraction_leaf:
            _refuse("min_weight_fraction_leaf",
                    "sample weights do not cross this boundary yet.")
        super().__init__(
            n_estimators, max_depth, max_leaf_nodes, max_features, n_bins,
            min_samples_leaf, min_samples_split, min_impurity_decrease,
            bootstrap, max_samples, random_state, n_streams, max_batch_size,
            oob_score, warm_start, ccp_alpha, None, monotonic_cst,
            n_jobs, verbose, device,
        )

    def fit(self, X, y):
        y32 = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
        return self._fit_arrays(X, y32, 0, _mojolearn_rf.rf_regressor_fit)

    def predict(self, X):
        Xa, n_rows, n_features = self._check_predict_input(X)
        out = np.empty(n_rows, dtype=np.float32)
        wrote = _mojolearn_rf.rf_predict_reg(
            _addr_ro(self._offsets),
            _addr_ro(self._colid),
            _addr_ro(self._quesval),
            _addr_ro(self._left_child),
            _addr_ro(self._leaves),
            _addr_ro(Xa),
            _addr(out),
            [int(n_rows), int(n_features), self._n_trees, 1],
        )
        if wrote != n_rows:
            raise RuntimeError(
                f"rf_predict_reg wrote {wrote} of {n_rows} rows"
            )
        return out
