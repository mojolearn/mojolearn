# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Random Forest on the GPU: cuML's forest, and cuML's defaults but one.

The learner is `ensemble/`: the port of cuML's `ML::RandomForest`
(`randomforest.cuh`) with its batched-levelalgo tree builder, quantile
binning, and the with-replacement `RowSampler` -- THIS wrapper is that
sampler's first Python caller (the `extratrees` surface refuses
`bootstrap=True` by name because its copy has no caller).

THE DEFAULTS ARE cuML's, NOT scikit-learn's, per the package rule
("the defaults follow the upstream each algorithm mirrors"), WITH ONE
NAMED EXCEPTION -- `max_depth`, DEVIATION 409, stated in full at the
constant below. The two that will surprise an sklearn user, documented
on the classes as well:

* `max_depth` defaults to 16 here, which is NEITHER the pinned cuML's
  default NOR sklearn's. It is the default cuML RETIRED in the very
  release this port pins. DEVIATION 409.
* splits are searched over at most `n_bins` (default 128) per-feature
  QUANTILES, cuML's design, not sklearn's exact thresholds. Faster, and a
  different algorithm -- a comparison against sklearn must say so.

EVERY sklearn-SHAPED PARAMETER IS EITHER HONOURED OR REFUSED BY NAME. None
is accepted and ignored.

`max_leaf_nodes` IS REFUSED, AND IT WAS SILENTLY ALIASED UNTIL 2026-09-01
(DEVIATION 408). sklearn's `max_leaf_nodes=k` selects BEST-FIRST growth to
exactly k leaves -- `tree/_classes.py:446-447` swaps in
`BestFirstTreeBuilder` the moment the value is not None -- and there is no
best-first grower in `ensemble/`. cuML's `max_leaves` is a CAP on a
LEVEL-ORDER grower, which answers a different question and returns a
different tree from the same data, so this wrapper now carries cuML's knob
under cuML's own name, `max_leaves=`, and refuses sklearn's spelling by
name. The best-first semantics DO exist in this library, on the
`extratrees` surface, whose `max_leaf_nodes` IS sklearn's (`extratrees/`,
DEVIATION BLOCKS 466 to 469). The two are deliberately not spelled alike on
either surface and are never aliased onto each other.

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

from . import _mojolearn_rf, _serialize
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-randomforest-1"

# DEVIATION 409 (2026-09-01, RF lane). THIS SURFACE'S `max_depth` DEFAULT
# IS NOT THE PINNED cuML'S, AND THE NAME IT USED TO CARRY SAID IT WAS.
# WHAT WAS HERE: `_CUML_DEFAULT_MAX_DEPTH = 16`, and a module docstring
# citing `randomforestclassifier.pyx` for it.
# WHY THAT IS WRONG, read out of the pin
# (`~/CascadeProjects/upstream/cuml-v26.08.00`, 265b9da6): 16 WAS cuML's
# default, and cuML retired it in THE VERY RELEASE THIS PORT PINS. Both
# estimators now default to `None` and both say so in their own words --
# `randomforestclassifier.py:68-74` and `randomforestregressor.py:62-68`
# read "max_depth : int or None (default = None)" followed by
# ".. versionchanged:: 26.08  The default of `max_depth` changed from
# `16` to `None`", and `randomforest_common.pyx:317` is `max_depth=None`
# in the shared `__init__`. `None` is not passed through: it is
# marshalled to `np.iinfo(np.int32).max` on the way to C++
# (`randomforest_common.pyx:480-481`), because the C++ header's own -1
# sentinel is a value their validator REFUSES (`ASSERT(params.max_depth
# >= 0)`, `decisiontree.cu:19`). So the pinned cuML default is UNLIMITED
# DEPTH, spelled INT32_MAX.
# The citation was also unresolvable at the pin: there is no
# `randomforestclassifier.pyx` in v26.08.00, only
# `randomforestclassifier.py` -- the same stale-citation class that
# `ensemble/randomforest.mojo:23-31` already names.
# THE MOJO SURFACE OF THE SAME LEARNER FOLLOWED THE PIN AND THIS ONE DID
# NOT: `ensemble/randomforest.mojo:1062` (`default_rf_params_classifier`)
# and `:1091` (`default_rf_params_regressor`) both pass
# `max_depth=INT32_MAX`, matching the table at
# `ensemble/randomforest.mojo:43`. Two entry points into one forest
# therefore disagree about what an unspecified depth means.
# WHAT IS STILL TRUE: 16 is what THIS wrapper ships. The value below is
# the shipped behaviour, so it is left where it is and renamed to stop
# claiming an upstream that does not say it. Flipping it to INT32_MAX is
# the alignment this deviation is owed and it is a BEHAVIOUR change --
# unspecified depth would grow to purity, deeper and slower trees from
# the same call -- so it is not made from a read. No recorded number
# depends on the implicit default: every bench and probe arm passes
# `max_depth` explicitly (`bench/speed/forest_speed_arm.py:230`,
# `tools/repeat_run_stability.py:117`), and the one bare
# `RandomForestClassifier()` construction in the tree
# (`python/mojolearn/tests/test_rf_leaf_budget.py:372`) only reaches
# `_bind` and never fits. The bare call in
# `tools/preprocess_identity_probe.py:15` is illustrative prose inside a
# docstring, not a call site.
# The other three constants below WERE re-read against the pin in the
# same pass and are correct: `randomforest_common.pyx:319` is
# `n_bins=128`, `:324` is `max_batch_size=4096`, `:326` is `n_streams=4`.
_PORT_DEFAULT_MAX_DEPTH = 16
_CUML_DEFAULT_N_BINS = 128
_CUML_DEFAULT_N_STREAMS = 4
_CUML_DEFAULT_MAX_BATCH = 4096

# DEVIATION 407 (2026-08-23, RF lane): the split criterion crosses the
# boundary as cuML's `CRITERION` enumerator (`algo_helper.h:10-18`, the
# same integers `ensemble/decisiontree/decisiontree.mojo` declares: GINI 0,
# ENTROPY 1, MSE 2, POISSON 4, GAMMA 5, INVERSE_GAUSSIAN 6). The names are
# sklearn's where sklearn has one and cuML's otherwise, mapped the way
# `randomforest_common.pyx:104-120` maps cuML's own strings; `log_loss` is
# sklearn's second spelling of entropy and `mse` is cuML's spelling of
# squared_error. The binding refuses by name any enumerator its objective
# has no arm for (`bindings/_mojolearn_rf.mojo::_check_criterion`), so a
# wrong code cannot fit a silent forest of stumps.
_CLS_CRITERIA = {"gini": 0, "entropy": 1, "log_loss": 1}
_REG_CRITERIA = {
    "squared_error": 2, "mse": 2,
    "poisson": 4, "gamma": 5, "inverse_gaussian": 6,
}
# sklearn names the engine enumerates but has no arm for. MAE is value 3
# of their enum and refused by `DecisionTreeParams.check()`
# (`ensemble/decisiontree/decisiontree.mojo:286-291`, after
# `decisiontree.cu:28` and `randomforest_common.pyx:147-151`);
# friedman_mse has no cuML counterpart at all.
_NOT_PORTED_CRITERIA = {
    "absolute_error": "cuML enumerates MAE (algo_helper.h:14) and refuses"
                      " it (decisiontree.cu:28); the port keeps that refuse"
                      " at ensemble/decisiontree/decisiontree.mojo:286.",
    "mae": "cuML enumerates MAE (algo_helper.h:14) and refuses it"
           " (decisiontree.cu:28); the port keeps that refuse at"
           " ensemble/decisiontree/decisiontree.mojo:286.",
    "friedman_mse": "no cuML counterpart; RegressionObjectiveFunction"
                    ".GainPerSplit (ensemble/decisiontree/batched_levelalgo"
                    "/objectives.mojo:1252-1261) has no arm for it.",
}


# DEVIATION 408 (2026-09-01, RF lane). SKLEARN'S LEAF BUDGET IS REFUSED
# RATHER THAN ALIASED.
# WHAT WAS HERE: `max_leaves=(-1 if max_leaf_nodes is None else
# int(max_leaf_nodes))`, a straight rename of one parameter onto another.
# WHY THAT IS WRONG: sklearn's `max_leaf_nodes=k` is a GROWTH ORDER, not a
# bound. `tree/_classes.py:446-447` reads "Use BestFirst if max_leaf_nodes
# given; use DepthFirst otherwise" and constructs `BestFirstTreeBuilder`,
# which expands the frontier node with the largest impurity improvement
# until exactly k leaves exist. cuML's `max_leaves` bounds a LEVEL-ORDER
# grower and reorders nothing: `NodeQueue::Pop` takes from the FRONT of a
# FIFO and `Push` appends to the BACK (`builder.cuh:70-78`, `:117`,
# transcribed at `ensemble/decisiontree/batched_levelalgo/builder.mojo`),
# and the budget is spent by whichever nodes that order reaches first --
# tested in `IsExpandable` (`builder.cuh:82-88`) and again inside `Push`
# (`:101`). Same k, different tree, no error and no warning.
# HOW LONG IT STOOD: the E2 cell `rf_clf_maxleaf` has scored IDENTICAL
# (388 stages) in every recorded round on every column, which certifies
# that the aliased answer is REPRODUCIBLE and says nothing about whether it
# is the answer that was asked for.
# THE FIX IS AT THIS BOUNDARY, not in the builder: cuML's knob is now
# carried under cuML's own name (`max_leaves=`, slot 5, unchanged), and
# sklearn's spelling is refused by name with a pointer at the surface that
# does implement it.
_MAX_LEAF_NODES_WHY = (
    "sklearn's max_leaf_nodes=k is BEST-FIRST GROWTH to exactly k leaves"
    " (tree/_classes.py:446-447 swaps in BestFirstTreeBuilder the moment"
    " the value is not None), and ensemble/ has no best-first grower."
    " cuML's nearest knob, max_leaves, is a CAP on a LEVEL-ORDER grower"
    " that reorders nothing -- NodeQueue pops from the front of a FIFO and"
    " pushes to the back (builder.cuh:70-78, :117) and the budget is spent"
    " by whichever nodes that order reaches first (:82-88, :101) -- so the"
    " same k gives a DIFFERENT TREE. This surface aliased one onto the"
    " other until 2026-09-01 and returned that different tree without"
    " saying so. Pass max_leaves= for cuML's cap, which these classes now"
    " carry under its own name, or use mojolearn.ExtraTreesClassifier /"
    " ExtraTreesRegressor, whose max_leaf_nodes IS sklearn's best-first"
    " budget (extratrees/, DEVIATION BLOCKS 466 to 469)."
)


def _refuse(name, why):
    raise NotImplementedError(
        f"{name} is not ported: {why} Refused by name rather than accepted"
        " and ignored."
    )


def _criterion_code(criterion, table, who):
    """A criterion NAME into the engine's `CRITERION` enumerator, or a
    refusal by name (DEVIATION 407)."""
    if criterion in table:
        return table[criterion]
    if criterion in _NOT_PORTED_CRITERIA:
        _refuse(f"criterion={criterion!r}", _NOT_PORTED_CRITERIA[criterion])
    raise ValueError(
        f"criterion={criterion!r} is not a {who} criterion here; accepted"
        f" are {sorted(table)}"
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


def _max_leaves_slot(max_leaves):
    """cuML's `max_leaves` into slot 5 of the params list.

    -1 is THEIR sentinel for unlimited, not ours (`decisiontree.hpp:83`,
    `randomforest_common.pyx:318`, and the default table in
    `ensemble/randomforest.mojo`); a positive int is a cap. `None` is
    accepted as a spelling of -1 because a Python caller reaches for it.
    Everything else is refused HERE rather than by
    `DecisionTreeParams.check()` (`ensemble/decisiontree/decisiontree.mojo`
    :257-258, which raises "Invalid max leaves"), so the message can name
    the sentinel the caller was supposed to pass.
    """
    if max_leaves is None:
        return -1
    if isinstance(max_leaves, bool) or not isinstance(
        max_leaves, (int, np.integer)
    ):
        raise ValueError(
            f"max_leaves={max_leaves!r} must be None, -1 (cuML's sentinel"
            " for unlimited) or a positive int count"
        )
    v = int(max_leaves)
    if v == -1 or v > 0:
        return v
    raise ValueError(
        f"max_leaves={v} is not a leaf cap: cuML accepts -1 (unlimited) or"
        " a positive count, and ensemble/decisiontree/decisiontree.mojo"
        ":257-258 raises on anything else"
    )


class _RandomForestBase(NumericModeMixin):
    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_rf"

    def __init__(
        self,
        n_estimators,
        max_depth,
        max_leaf_nodes,
        max_leaves,
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
        criterion_code,
    ):
        if device != "gpu":
            raise ValueError(
                f"device must be 'gpu', got {device!r}: ensemble/ has no"
                " host transcription of the forest builder"
            )
        if max_leaf_nodes is not None:
            _refuse("max_leaf_nodes", _MAX_LEAF_NODES_WHY)
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
            criterion=int(criterion_code),
            n_estimators=int(n_estimators),
            max_depth=(
                _PORT_DEFAULT_MAX_DEPTH if max_depth is None
                else int(max_depth)
            ),
            max_leaves=_max_leaves_slot(max_leaves),
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
        out = fit_fn(
            _addr_ro(Xf), _addr_ro(y_arr), params, self._cfg["criterion"]
        )
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

    def save(self, path):
        """Write the fitted forest to `path` as an npz.

        The file holds the five prediction arrays exactly as fitted, raw
        bytes and exact dtypes, so a model saved on one machine and loaded
        on another predicts the SAME BITS. Floats never pass through
        decimal text. The bytes of the file itself are a pure function of
        the model (see `_serialize.write_npz`), so equal models give equal
        file hashes across machines.
        """
        if not hasattr(self, "_offsets"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": np.asarray(_MODEL_FORMAT),
            "estimator": np.asarray(type(self).__name__),
            "device": np.asarray(self.device),
            "offsets": self._offsets,
            "colid": self._colid,
            "quesval": self._quesval,
            "left_child": self._left_child,
            "leaves": self._leaves,
            "meta": np.asarray(
                [
                    self.n_features_in_,
                    self._n_trees,
                    self._num_outputs,
                ],
                dtype=np.int64,
            ),
        }
        if hasattr(self, "classes_"):
            arrays["classes"] = np.asarray(self.classes_)
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a forest saved by `save`. The result predicts; it does not
        refit. Loading a file saved by the other estimator class raises
        rather than reinterpreting its leaves."""
        arrays = _serialize.read_npz(path, _MODEL_FORMAT)
        saved_as = _serialize.scalar_str(arrays, "estimator")
        if saved_as != cls.__name__:
            raise ValueError(
                f"mojolearn: {path!r} was saved by {saved_as}, not "
                f"{cls.__name__}"
            )
        obj = cls.__new__(cls)
        obj.device = _serialize.scalar_str(arrays, "device")
        obj._offsets = _serialize.exact(arrays, "offsets", np.int32)
        obj._colid = _serialize.exact(arrays, "colid", np.int32)
        obj._quesval = _serialize.exact(arrays, "quesval", np.float32)
        obj._left_child = _serialize.exact(arrays, "left_child", np.int32)
        obj._leaves = _serialize.exact(arrays, "leaves", np.float32)
        meta = _serialize.exact(arrays, "meta", np.int64)
        obj.n_features_in_ = int(meta[0])
        obj._n_trees = int(meta[1])
        obj._num_outputs = int(meta[2])
        if "classes" in arrays:
            obj.classes_ = arrays["classes"]
            obj.n_classes_ = int(len(obj.classes_))
        return obj


class RandomForestClassifier(_RandomForestBase):
    """cuML's `RandomForestClassifier`, honoured or refused by name.

    `criterion` is 'gini' (cuML's GINI, the default) or 'entropy' /
    'log_loss' (ENTROPY), split over per-feature quantiles (`n_bins`,
    default 128); `max_depth` defaults to 16, which is neither the pinned
    cuML's default nor sklearn's -- cuML changed its own from 16 to `None`
    in v26.08.00, the release this port pins (DEVIATION 409).
    `predict` is the argmax of `predict_proba`, exactly as
    `RandomForest::predict` argmaxes (`randomforest.cuh:417-427`).
    Entropy's `log` is DEVIATION 113/406's: it runs in both numeric modes
    and is bit-comparable to cuML's in neither.

    `max_leaves` is cuML's leaf budget, -1 (unlimited) by default, and it
    caps a LEVEL-ORDER grower. It is NOT sklearn's `max_leaf_nodes`, which
    is best-first growth to exactly k leaves and is refused by name here
    (DEVIATION 408); `mojolearn.ExtraTreesClassifier` is where that lives.
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
        max_leaves=-1,
        n_bins=_CUML_DEFAULT_N_BINS,
        n_streams=_CUML_DEFAULT_N_STREAMS,
        max_batch_size=_CUML_DEFAULT_MAX_BATCH,
        device="gpu",
    ):
        code = _criterion_code(criterion, _CLS_CRITERIA, "classifier")
        if min_weight_fraction_leaf:
            _refuse("min_weight_fraction_leaf",
                    "sample weights do not cross this boundary yet.")
        super().__init__(
            n_estimators, max_depth, max_leaf_nodes, max_leaves,
            max_features, n_bins,
            min_samples_leaf, min_samples_split, min_impurity_decrease,
            bootstrap, max_samples, random_state, n_streams, max_batch_size,
            oob_score, warm_start, ccp_alpha, class_weight, monotonic_cst,
            n_jobs, verbose, device, code,
        )
        self.criterion = criterion

    def fit(self, X, y):
        ya = np.asarray(y).ravel()
        self.classes_, codes = np.unique(ya, return_inverse=True)
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("y has fewer than 2 classes")
        y32 = np.ascontiguousarray(codes, dtype=np.int32)
        return self._fit_arrays(
            X, y32, self.n_classes_, self._bind("_mojolearn_rf").rf_classifier_fit
        )

    def predict_proba(self, X):
        Xa, n_rows, n_features = self._check_predict_input(X)
        out = np.empty(n_rows * self._num_outputs, dtype=np.float32)
        wrote = self._bind("_mojolearn_rf").rf_predict_proba(
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
    """cuML's `RandomForestRegressor`, honoured or refused by name.

    `criterion` is 'squared_error' (cuML's MSE, the default; 'mse' is
    accepted as cuML's spelling), 'poisson', 'gamma' or 'inverse_gaussian'
    -- the four arms of their `GainPerSplit` (`objectives.cuh:331-338`).
    'absolute_error' / 'mae' and 'friedman_mse' are refused by name
    (DEVIATION 407). Same quantile-split default as the classifier, and the
    same `max_depth=16`, which is this surface's and NOT the pinned cuML's
    -- theirs became `None` in v26.08.00 (DEVIATION 409). `max_features`
    defaults to 1.0, which IS cuML's regressor default
    (`randomforestregressor.py:162`).

    TARGET DOMAIN. The three deviance criteria return `-max()` for any
    candidate whose label sum is not positive (`objectives.cuh:251-253`,
    `:279-281`, `:306-308`), so cuML, fed a non-positive target, silently
    fits stumps. `fit` refuses instead: 'poisson' needs y >= 0 with a
    positive sum (sklearn's rule, which is the engine's per-node guard
    applied at the root); 'gamma' and 'inverse_gaussian' need y > 0.

    `max_leaves` is cuML's leaf budget, -1 (unlimited) by default, and it
    caps a LEVEL-ORDER grower. It is NOT sklearn's `max_leaf_nodes`, which
    is best-first growth to exactly k leaves and is refused by name here
    (DEVIATION 408); `mojolearn.ExtraTreesRegressor` is where that lives.
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
        max_leaves=-1,
        n_bins=_CUML_DEFAULT_N_BINS,
        n_streams=_CUML_DEFAULT_N_STREAMS,
        max_batch_size=_CUML_DEFAULT_MAX_BATCH,
        device="gpu",
    ):
        code = _criterion_code(criterion, _REG_CRITERIA, "regressor")
        if min_weight_fraction_leaf:
            _refuse("min_weight_fraction_leaf",
                    "sample weights do not cross this boundary yet.")
        super().__init__(
            n_estimators, max_depth, max_leaf_nodes, max_leaves,
            max_features, n_bins,
            min_samples_leaf, min_samples_split, min_impurity_decrease,
            bootstrap, max_samples, random_state, n_streams, max_batch_size,
            oob_score, warm_start, ccp_alpha, None, monotonic_cst,
            n_jobs, verbose, device, code,
        )
        self.criterion = criterion

    def fit(self, X, y):
        y32 = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
        code = self._cfg["criterion"]
        if code == _REG_CRITERIA["poisson"]:
            if np.any(y32 < 0) or not np.sum(y32, dtype=np.float64) > 0:
                raise ValueError(
                    "criterion='poisson' requires y >= 0 with a positive"
                    " sum: PoissonGain returns -max() for a non-positive"
                    " label sum (objectives.cuh:251-253), which would fit"
                    " a stump silently"
                )
        elif code in (_REG_CRITERIA["gamma"],
                      _REG_CRITERIA["inverse_gaussian"]):
            if np.any(y32 <= 0):
                raise ValueError(
                    f"criterion={self.criterion!r} requires y > 0: its"
                    " gain returns -max() for a non-positive label sum"
                    " (objectives.cuh:279-281, :306-308), which would fit"
                    " a stump silently"
                )
        return self._fit_arrays(X, y32, 0, self._bind("_mojolearn_rf").rf_regressor_fit)

    def predict(self, X):
        Xa, n_rows, n_features = self._check_predict_input(X)
        out = np.empty(n_rows, dtype=np.float32)
        wrote = self._bind("_mojolearn_rf").rf_predict_reg(
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
