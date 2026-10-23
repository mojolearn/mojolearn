# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Random Forest on the GPU: cuML's forest, and cuML's defaults.

The learner is `ensemble/`: a random forest (reference: cuML's `ML::RandomForest`,
`randomforest.cuh`) with a batched-levelalgo tree builder, quantile
binning, and the with-replacement `RowSampler` -- THIS wrapper is that
sampler's first Python caller (the `extratrees` surface refuses
`bootstrap=True` by name because its copy has no caller).

THE DEFAULTS ARE cuML's, NOT scikit-learn's, per the package rule
("the defaults follow each algorithm's reference") -- with no
exception since 2026-09-01, when DEVIATION 409 (this surface shipped
cuML's RETIRED pre-26.08 `max_depth=16`) was CLOSED by aligning; full
history at the constant below. Two notes for an sklearn user,
documented on the classes as well:

* `max_depth` defaults to `None`, UNLIMITED depth -- `None` maps to
  `np.iinfo(np.int32).max` exactly as the pinned cuML marshals it
  (`randomforest_common.pyx:480-481`). That matches both cuML (whose own
  default changed from 16 to `None` in v26.08.00, the release this implementation
  pins) and sklearn. Until 2026-09-01 an unspecified depth meant 16
  here; pass `max_depth=16` to keep that behaviour (DEVIATION 409).
* splits are searched over at most `n_bins` (default 128) per-feature
  QUANTILES, as in cuML, not sklearn's exact thresholds. Faster, and a
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

X IS COPIED TWICE PER FIT (the host to column-major float32 through
`_buffer.as_f32_colmajor`, zero-copy for a float32 F-order input, then
across the boundary); on large matrices that is the dominant cost of the
CALL.

INPUTS AND OUTPUTS ARE NOT NumPy (DEVIATION 2341). `X` is anything the
buffer protocol exposes -- an ndarray, an `array.array`, a
`mojolearn.Array` -- or a nested list; `y` for the regressor likewise, and
for the classifier any sequence of label objects. `predict_proba` and the
regressor's `predict` return a `mojolearn.Array` (`numpy.asarray(result)`
is a zero-copy view for a caller who has NumPy). `classes_` is a PYTHON
LIST of the caller's label objects in the order `_labels.py` defines
(DEVIATION 2340: numbers by value, strings by code point, NaN refused),
and the classifier's `predict` returns an int64 or float64 `Array` for
numeric labels and a Python list for str labels. Nothing in this module
imports NumPy.
"""

from . import _portable_math as math
import numbers

from . import _mojolearn_rf, _native_archive_bridge, _native_trees_bridge
from ._labels import argmax_rows, decode_labels, is_bool
from ._mode import NumericModeMixin
from ._forest_protocol import ForestProtocol, forest_estimator

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-randomforest-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered after `_RandomForestBase`;
# bindings/host_archive_native.mojo `host_archive_public_forest` builds the
# loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

# The compiled tree-ensemble owner, connected once at import: `fit`,
# `predict` and `predict_proba` are one native call each
# (bindings/trees_public_forest.mojo validates the parameters, encodes the
# labels, checks the weights, picks the fit entry and inference engine,
# launches the kernels and assigns the fitted state).
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()

# DEVIATION 409 (2026-09-01, RF lane). CLOSED -- ALIGNED the same day;
# the closure paragraph is below, after the history. AS FOUND: this
# surface's `max_depth` default WAS NOT the pinned cuML's, and the name
# it used to carry said it was.
# WHAT WAS HERE: `_CUML_DEFAULT_MAX_DEPTH = 16`, and a module docstring
# citing `randomforestclassifier.pyx` for it.
# WHY THAT IS WRONG, read out of the pin
# (`~/CascadeProjects/upstream/cuml-v26.08.00`, 265b9da6): 16 WAS cuML's
# default, and cuML retired it in THE VERY RELEASE THIS IMPLEMENTATION PINS. Both
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
# therefore disagreed about what an unspecified depth means.
# CLOSED -- ALIGNED by decision on 2026-09-01. The history above stands: the wrapper shipped 16 (first
# under a name claiming it was cuML's, then, from the morning of
# 2026-09-01, honestly as a named deviation) and the value below was 16
# until the alignment later that same day. It is now 2147483647,
# `np.iinfo(np.int32).max` spelled as the literal (DEVIATION 2341, the
# NumPy-free boundary), the exact value the pinned cuML substitutes
# for `None` (`randomforest_common.pyx:480-481`), so the constant
# MATCHES the pin and both surfaces of the learner give one answer for
# an unspecified depth: UNLIMITED, grow to purity. This IS a behaviour
# change -- deeper and slower trees from the same bare call; pass
# `max_depth=16` for the old behaviour (CHANGELOG.md, Unreleased). No
# recorded number depends on the implicit default: every bench and probe
# arm passes `max_depth` explicitly (`bench/speed/forest_speed_arm.py:230`,
# `tools/repeat_run_stability.py:117`), and the one bare
# `RandomForestClassifier()` construction in the tree
# (`python/mojolearn/tests/test_rf_leaf_budget.py:372`) only reaches
# `_bind` and never fits. The bare call in
# `tools/preprocess_identity_probe.py:15` is illustrative prose inside a
# docstring, not a call site. KNOWN-INERT:
# `bindings/_mojolearn_rf.mojo:472` still carries a fourth spelling of
# 16 on the predict path, inert because predict reads the fitted
# forest's trees, not these knobs. UNVERIFIED, RUN OWED (the flip was
# applied from a read): rebuild `_mojolearn_rf` on all tiers, then
# `python -m pytest python/mojolearn/tests/ -k rf` and
# `packaging/macos/smoke.py` on all tiers; the E2 RF identity cells
# re-run at the next rented leg, since unspecified-depth fits now grow
# deeper.
# The other three constants below WERE re-read against the pin in the
# same pass and are correct: `randomforest_common.pyx:319` is
# `n_bins=128`, `:324` is `max_batch_size=4096`, `:326` is `n_streams=4`.
_PORT_DEFAULT_MAX_DEPTH = 2147483647
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
_UNSUPPORTED_CRITERIA = {
    "absolute_error": "cuML enumerates MAE (algo_helper.h:14) and refuses"
                      " it (decisiontree.cu:28); the implementation keeps that refuse"
                      " at ensemble/decisiontree/decisiontree.mojo:286.",
    "mae": "cuML enumerates MAE (algo_helper.h:14) and refuses it"
           " (decisiontree.cu:28); the implementation keeps that refuse at"
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
# matched at `ensemble/decisiontree/batched_levelalgo/builder.mojo`),
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


_NONFINITE_REFUSAL = "X contains NaN or infinity; the forest has no missing-value arm"


def _refuse(name, why):
    raise NotImplementedError(
        f"{name} is not implemented: {why} Refused by name rather than accepted"
        " and ignored."
    )


def _min_weight_fraction(value, who):
    """scikit-learn's `min_weight_fraction_leaf` check, its wording
    (`Interval(Real, 0.0, 0.5, closed="both")`). The bound itself is
    applied by the fit binding (slot 16 of `_fit_params`)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        f = None
    if isinstance(value, bool) or f is None or not 0.0 <= f <= 0.5:
        raise ValueError(
            f"The 'min_weight_fraction_leaf' parameter of {who} must be a float"
            f" in the range [0.0, 0.5]. Got {value!r} instead.")
    return f


def _criterion_code(criterion, table, who):
    """A criterion NAME into the engine's `CRITERION` enumerator, or a
    refusal by name (DEVIATION 407)."""
    if criterion in table:
        return table[criterion]
    if criterion in _UNSUPPORTED_CRITERIA:
        _refuse(f"criterion={criterion!r}", _UNSUPPORTED_CRITERIA[criterion])
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
            # The owned host helper uses the correctly rounded CPU sqrt instruction.
            return math.sqrt(n_features) / n_features
        if max_features == "log2":
            # Use the same pinned portable_log2_64 arithmetic as the Mojo
            # builder. Keep the fraction: the kernel truncates after FP32
            # conversion. Powers of two are exact; max(2, n) retains the
            # existing nonzero fraction for a one-feature input.
            return math.log2(max(2, n_features)) / n_features
        raise ValueError(
            f"max_features={max_features!r} is not a recognised form;"
            " accepted are 'sqrt', 'log2', None, a float fraction or an"
            " int count"
        )
    if isinstance(max_features, numbers.Integral) and not is_bool(
        max_features
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
    if is_bool(max_leaves) or not isinstance(
        max_leaves, numbers.Integral
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


class _RandomForestBase(ForestProtocol, NumericModeMixin):
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
                f"device must be 'gpu', got {device!r}: the fit runs on the"
                " GPU binding, or, on a CPU-only install with the rf host"
                " binding built, on its host restatement"
                " (ensemble/host/rf_oracle.mojo) under the same device='gpu'"
            )
        if max_leaf_nodes is not None:
            _refuse("max_leaf_nodes", _MAX_LEAF_NODES_WHY)
        if not is_bool(oob_score):
            raise ValueError("oob_score must be a boolean; callable scoring is not supported")
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")
        if warm_start:
            _refuse("warm_start=True", "there is no incremental fit here,"
                    " and accepting it would silently refit from scratch.")
        if ccp_alpha:
            _refuse("ccp_alpha", "cost-complexity pruning is a"
                    " post-processing pass neither cuML nor this implementation"
                    " implements. Only 0.0 is accepted.")
        if class_weight is not None and not isinstance(class_weight, dict):
            if not isinstance(class_weight, str) or class_weight != "balanced":
                raise ValueError("class_weight must be None, a dict, or 'balanced'; balanced_subsample is unsupported")
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
            oob_score=bool(oob_score),
        )

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


#: The saved formats `load` reads, and the save schema of every forest
#: class saved in this format (bindings/archive_schema_native.mojo): the
#: members, dtypes and refusals the 0.8.55 Python `save` wrote.
_MODEL_FORMATS = (_MODEL_FORMAT, _MODEL_FORMAT + "-parallel-groves-1")
_SAVE_SCHEMA = (
    _native_archive_bridge.fitted("_offsets"),
    ("format", "format", _MODEL_FORMAT),
    ("estimator", "estimator"),
    ("array", "device", "device"),
    ("array", "offsets", "_offsets"),
    ("array", "colid", "_colid"),
    ("array", "quesval", "_quesval"),
    ("array", "left_child", "_left_child"),
    ("array", "leaves", "_leaves"),
    ("i8", "meta", (("int", "n_features_in_"), ("int", "_n_trees"), ("int", "_num_outputs"))),
    ("classes", "classes", "classes_"),
    ("forest_engine", _MODEL_FORMAT),
)
_native_archive_bridge.register(_RandomForestBase, _MODEL_FORMATS, _SAVE_SCHEMA)


# Maintainer notes (moved from the docstring): criteria cross as cuML's
# CRITERION enumerator (DEVIATION 407). `predict` argmaxes exactly as
# `RandomForest::predict` (`randomforest.cuh:417-427`). Entropy's `log` is
# DEVIATION 113/406's: it runs in both numeric modes and is bit-comparable to
# cuML's in neither. `max_depth=None` matches cuML >= v26.08.00; this surface
# substituted 16 until 2026-09-01 (DEVIATION 409, closed by aligning).
# `max_leaf_nodes` is refused by name (DEVIATION 408, see _MAX_LEAF_NODES_WHY).
@forest_estimator("classifier")
class RandomForestClassifier(_RandomForestBase):
    """Random forest classifier with cuML's defaults.

    An ensemble of bootstrapped decision trees whose split thresholds are
    searched over at most `n_bins` per-feature quantiles, as in cuML, rather
    than scikit-learn's exact thresholds. Defaults follow cuML. Every
    scikit-learn-shaped parameter is either honoured or refused by name with
    `NotImplementedError`; none is accepted and ignored.

    `X` is converted to float32; NaN or infinity in `X` raises `ValueError`.
    `y` may hold numbers or strings; NaN labels are refused. The fit is
    reproducible from `random_state`, and `random_state=None` means seed 0
    (deterministic), unlike scikit-learn. Supports the `fast` (Apple GPUs only)
    and `identical` numeric modes through `numeric_mode`.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    criterion : {'gini', 'entropy', 'log_loss'}, default 'gini'
        Split criterion; 'log_loss' is a second spelling of 'entropy'.
    max_depth : int or None, default None
        Maximum tree depth. None means unlimited, as in cuML and scikit-learn.
    min_samples_split : int, default 2
        Minimum rows needed to split a node.
    min_samples_leaf : int, default 1
        Minimum rows in a leaf.
    min_weight_fraction_leaf : float, default 0.0
        Minimum fraction of the total sample weight (of all rows when no
        weights are given) required in each leaf, in [0.0, 0.5].
    max_features : {'sqrt', 'log2'}, int, float or None, default 'sqrt'
        Features considered per split: a name, a count in [1, n_features],
        a fraction in (0, 1], or None for all features.
    max_leaf_nodes : None, default None
        Refused by name: scikit-learn's best-first growth to exactly k leaves
        is not available here. Use `max_leaves` for cuML's level-order leaf
        cap, or `mojolearn.ExtraTreesClassifier`, whose `max_leaf_nodes` is
        scikit-learn's.
    min_impurity_decrease : float, default 0.0
        Minimum impurity decrease required to split.
    bootstrap : bool, default True
        Sample rows with replacement for each tree.
    oob_score : bool, default False
        Compute the out-of-bag score during `fit`; requires
        `bootstrap=True`. Callable scorers are not supported.
    n_jobs : None, default None
        Only None is accepted.
    random_state : int or None, default None
        Seed; None means 0.
    verbose : int, default 0
        Only 0 is accepted.
    warm_start : bool, default False
        Only False is accepted; there is no incremental fit.
    class_weight : dict, 'balanced' or None, default None
        Per-class weights. 'balanced_subsample' is not supported.
    ccp_alpha : float, default 0.0
        Only 0.0 is accepted; cost-complexity pruning is not implemented.
    max_samples : float or None, default None
        Fraction of rows drawn per tree, in (0, 1]. None means all rows.
    monotonic_cst : None, default None
        Only None is accepted.
    max_leaves : int, default -1
        cuML's leaf cap on a level-order grower; -1 (or None) means
        unlimited. This is not scikit-learn's `max_leaf_nodes`.
    n_bins : int, default 128
        Number of per-feature quantile bins searched for splits.
    n_streams : int, default 4
        cuML's stream count parameter.
    max_batch_size : int, default 4096
        cuML's node batch size for the tree builder.
    device : {'gpu'}, default 'gpu'
        Only 'gpu' is accepted. On a CPU-only install the same setting runs
        the host build of the forest.
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
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen in `fit`.
    oob_score_ : float
        Out-of-bag score; present when `oob_score=True`.
    oob_decision_function_ : Array
        Out-of-bag class probabilities, `(n_samples, n_classes)` float64;
        present when `oob_score=True`.
    feature_importances_ : Array
        Impurity importances, float32; present when `fit` was called with
        `feature_importances=True`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] > 0.5).astype(np.int64)
    >>> model = mojolearn.RandomForestClassifier(n_estimators=50).fit(X, y)
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
        inference_engine="auto",
    ):
        code = _criterion_code(criterion, _CLS_CRITERIA, "classifier")
        mwf = _min_weight_fraction(min_weight_fraction_leaf, type(self).__name__)
        super().__init__(
            n_estimators, max_depth, max_leaf_nodes, max_leaves,
            max_features, n_bins,
            min_samples_leaf, min_samples_split, min_impurity_decrease,
            bootstrap, max_samples, random_state, n_streams, max_batch_size,
            oob_score, warm_start, ccp_alpha, class_weight, monotonic_cst,
            n_jobs, verbose, device, code,
        )
        self.criterion = criterion
        self._cfg["min_weight_fraction_leaf"] = mwf

    def fit(self, X, y, sample_weight=None, *, feature_importances=False):
        """Fit the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,), default None
            Per-row weights, non-negative and not all zero. Multiplied into the
            ``class_weight`` row weights.
        feature_importances : bool, default False
            Also compute `feature_importances_` during this fit.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, feature_importances, None), _TREES_CONTEXT)

    def _fit_with_tree_start(self, X, y, tree_start=None, *, feature_importances=False,
                             sample_weight=None):
        """`fit` of the trees from global tree ID `tree_start` on (a shard of
        a multi-device forest)."""
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, feature_importances, tree_start),
                      _TREES_CONTEXT)

    def predict_proba(self, X):
        """Class probabilities: the per-class vote fractions over the trees.

        Returns
        -------
        Array of shape (n_samples, n_classes), float32
            Columns in `classes_` order.
        """
        return _TREES(self, "trees_forest_predict_proba", (X,), _TREES_CONTEXT)

    def _predict_with_proba(self, X):
        """Return both classifier views after one forest traversal.

        This is an internal verification fast path.  ``predict`` is defined
        as the first-max decode of ``predict_proba`` below, so retaining the
        exact probability buffer and deriving the labels from it is byte for
        byte the same work as the two public calls without traversing every
        tree twice.
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


# Maintainer notes (moved from the docstring): the regression criteria are
# the four arms of cuML's `GainPerSplit` (`objectives.cuh:331-338`);
# 'absolute_error'/'mae' and 'friedman_mse' are refused (DEVIATION 407).
# The three deviance criteria return `-max()` for a non-positive label sum
# (`objectives.cuh:251-253`, `:279-281`, `:306-308`), so cuML silently fits
# stumps; `fit` refuses instead. `max_features=1.0` is cuML's regressor
# default (`randomforestregressor.py:162`). `max_depth=None` since
# 2026-09-01 (DEVIATION 409); `max_leaf_nodes` refused (DEVIATION 408).
@forest_estimator("regressor")
class RandomForestRegressor(_RandomForestBase):
    """Random forest regressor with cuML's defaults.

    An ensemble of bootstrapped regression trees whose split thresholds are
    searched over at most `n_bins` per-feature quantiles, as in cuML, rather
    than scikit-learn's exact thresholds. Defaults follow cuML, including
    `max_features=1.0`. Every scikit-learn-shaped parameter is either
    honoured or refused by name with `NotImplementedError`.

    `X` and `y` are converted to float32; NaN or infinity in `X` raises
    `ValueError`. `random_state=None` means seed 0 (deterministic), unlike
    scikit-learn. Supports the `fast` (Apple GPUs only) and `identical`
    numeric modes through `numeric_mode`.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    criterion : {'squared_error', 'mse', 'poisson', 'gamma', 'inverse_gaussian'}, default 'squared_error'
        Split criterion; 'mse' is cuML's spelling of 'squared_error'.
        'poisson' requires finite `y >= 0` with at least one positive value;
        'gamma' and 'inverse_gaussian' require finite `y > 0`; otherwise
        `fit` raises `ValueError`. 'absolute_error', 'mae' and
        'friedman_mse' raise `NotImplementedError`.
    max_depth : int or None, default None
        Maximum tree depth. None means unlimited.
    min_samples_split : int, default 2
        Minimum rows needed to split a node.
    min_samples_leaf : int, default 1
        Minimum rows in a leaf.
    min_weight_fraction_leaf : float, default 0.0
        Minimum fraction of the total sample weight (of all rows when no
        weights are given) required in each leaf, in [0.0, 0.5].
    max_features : {'sqrt', 'log2'}, int, float or None, default 1.0
        Features considered per split: a name, a count, a fraction in
        (0, 1], or None for all features.
    max_leaf_nodes : None, default None
        Refused by name. Use `max_leaves` for cuML's level-order leaf cap, or
        `mojolearn.ExtraTreesRegressor`, whose `max_leaf_nodes` is
        scikit-learn's best-first budget.
    min_impurity_decrease : float, default 0.0
        Minimum impurity decrease required to split.
    bootstrap : bool, default True
        Sample rows with replacement for each tree.
    oob_score : bool, default False
        Compute the out-of-bag score during `fit`; requires `bootstrap=True`.
    n_jobs : None, default None
        Only None is accepted.
    random_state : int or None, default None
        Seed; None means 0.
    verbose : int, default 0
        Only 0 is accepted.
    warm_start : bool, default False
        Only False is accepted.
    ccp_alpha : float, default 0.0
        Only 0.0 is accepted.
    max_samples : float or None, default None
        Fraction of rows drawn per tree, in (0, 1]. None means all rows.
    monotonic_cst : None, default None
        Only None is accepted.
    max_leaves : int, default -1
        cuML's leaf cap on a level-order grower; -1 (or None) means
        unlimited.
    n_bins : int, default 128
        Number of per-feature quantile bins searched for splits.
    n_streams : int, default 4
        cuML's stream count parameter.
    max_batch_size : int, default 4096
        cuML's node batch size for the tree builder.
    device : {'gpu'}, default 'gpu'
        Only 'gpu' is accepted. On a CPU-only install the same setting runs
        the host build of the forest.
    inference_engine : {'auto', 'sequential', 'parallel_groves'}, default 'auto'
        How `predict` reduces over trees. 'auto' uses 'parallel_groves' in
        `fast` mode and 'sequential' otherwise.
    numeric_mode : {'fast', 'identical'} or None, default None
        Numeric mode for this estimator; None uses the package default.
        ``'fast'`` is available on Apple GPUs only.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in `fit`.
    oob_score_ : float
        Out-of-bag score; present when `oob_score=True`.
    oob_prediction_ : Array
        Out-of-bag predictions, float64; present when `oob_score=True`.
    feature_importances_ : Array
        Impurity importances, float32; present when `fit` was called with
        `feature_importances=True`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X[:, 0] * 2.0 + X[:, 1]
    >>> model = mojolearn.RandomForestRegressor(n_estimators=50).fit(X, y)
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
        inference_engine="auto",
    ):
        code = _criterion_code(criterion, _REG_CRITERIA, "regressor")
        mwf = _min_weight_fraction(min_weight_fraction_leaf, type(self).__name__)
        super().__init__(
            n_estimators, max_depth, max_leaf_nodes, max_leaves,
            max_features, n_bins,
            min_samples_leaf, min_samples_split, min_impurity_decrease,
            bootstrap, max_samples, random_state, n_streams, max_batch_size,
            oob_score, warm_start, ccp_alpha, None, monotonic_cst,
            n_jobs, verbose, device, code,
        )
        self.criterion = criterion
        self._cfg["min_weight_fraction_leaf"] = mwf

    def fit(self, X, y, sample_weight=None, *, feature_importances=False):
        """Fit the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,), default None
            Per-row weights, non-negative and not all zero.
        feature_importances : bool, default False
            Also compute `feature_importances_` during this fit.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, feature_importances, None), _TREES_CONTEXT)

    def _fit_with_tree_start(self, X, y, tree_start=None, *, feature_importances=False,
                             sample_weight=None):
        """`fit` of the trees from global tree ID `tree_start` on (a shard of
        a multi-device forest)."""
        return _TREES(self, "trees_forest_fit", (X, y, sample_weight, feature_importances, tree_start),
                      _TREES_CONTEXT)

    def predict(self, X):
        """Predicted values: the mean of the trees' predictions.

        Returns
        -------
        Array of shape (n_samples,), float32
        """
        return _TREES(self, "trees_forest_predict", (X,), _TREES_CONTEXT)


# The forests' native owner (bindings/trees_public_forest.mojo) reads these
# fixed tables: the criterion names, the refusal texts and the deviance
# criterion codes whose targets it checks.
_RF_TABLES = dict(
    cls_criteria=_CLS_CRITERIA, reg_criteria=_REG_CRITERIA, unsupported=_UNSUPPORTED_CRITERIA,
    max_leaf_nodes_why=_MAX_LEAF_NODES_WHY, poisson=_REG_CRITERIA["poisson"],
    gamma=_REG_CRITERIA["gamma"], inverse_gaussian=_REG_CRITERIA["inverse_gaussian"],
)
_native_trees_bridge.register(RandomForestClassifier, "rf_classifier", trees_rf_tables=_RF_TABLES)
_native_trees_bridge.register(RandomForestRegressor, "rf_regressor", trees_rf_tables=_RF_TABLES)
