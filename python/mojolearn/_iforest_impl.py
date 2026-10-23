# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Isolation Forest on the GPU. Reference: cuML's `IsolationForest`.

PRIVATE MODULE. `IsolationForest` is named exactly as scikit-learn names
it, but nothing here is re-exported from `mojolearn/__init__.py`; that
file is somebody else's and wiring this in is their call.

THE LANE'S STANDING, PLAINLY: `isolation_forest/` HAS NO CROSS-VENDOR
CARD. It is not in `tools/e1_bootstrap.sh` phase 8, not in
`tools/e3_round_judge.sh` section 7, and there is no leg artifact for it
anywhere under `bench/results/e1/`. `isolation_forest/README.md`'s "What
is owed" says it in its own words: "NVIDIA and AMD legs. Everything here
is Apple M4 only." Its nine gates are green on one M4 under IDENTICAL and
device equals the host oracle on 37,496 cells, which is a real property
and not the cross-vendor one. Nothing on this class may say otherwise.

Two more things the lane says about itself and this class inherits:

  * DEVIATION 750 is OPEN. cuML's `curand_u64` builds a 64-bit draw out of
    two unsequenced `curand()` calls, and C++ does not say which becomes
    the high word. Both readings conform and they give DIFFERENT FORESTS
    from the same seed. This implementation takes the first draw as the high word,
    by name, and that choice has never been checked against a cuML binary.
    Until it is, agreement with cuML is a belief, not a measurement. It is
    one number off a real NVIDIA GPU to close.
  * The card stages added on 2026-08-24 (`split.bounds`, `split.choice`,
    `rng.final`) HAD NOT BEEN COMPILED when this wrapper was written.

INPUTS AND OUTPUTS ARE NOT NumPy (DEVIATION 2344). `X` is anything the
buffer protocol exposes -- an ndarray, an `array.array`, a
`mojolearn.Array` -- or a nested list, converted once by
`_buffer.as_f32_c` (zero-copy for float32 C-order). `score_samples` and
`decision_function` return a float32 `mojolearn.Array`, `predict` an int32
one; `numpy.asarray(result)` is a zero-copy view for a caller who has
NumPy. Nothing in this module imports NumPy.
"""

import numbers

from . import _native_archive_bridge, _native_trees_bridge
from ._mode import NumericModeMixin


#: The model file (the neighbors and density inference lane, 2026-09-15).
#: DEVIATION 874 makes the fitted state the training matrix plus the
#: constructor's resolved knobs, so that is what `save` writes: `x` `<f4` as
#: fitted, `ints` `<i8` [n_estimators, max_samples_mode, max_samples_int,
#: max_depth, max_features_mode, max_features_int, bootstrap, seed,
#: contamination_auto, n_features_in_, max_samples_], `reals` `<f8`
#: [max_samples_frac, max_features_frac, contamination, offset_].
#: `mojolearn.host_model(path)` scores it on a CPU through the svm host
#: binding's iforest_run, the same rebuild every GPU scoring call runs.
_IFOREST_FORMAT = "mojolearn-iforest-1"
_IFOREST_INTS = 11
_IFOREST_REALS = 4

# The compiled owners, connected once at import: `fit`, `score_samples`,
# `decision_function`, `predict` and `fit_predict` are one call each into the
# tree owner (bindings/trees_public_iforest.mojo); `save` and `load` one call
# each into the saved-model owner.
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()


# Maintainer notes (moved from the docstring): the GPU implementation is
# `isolation_forest/` (DEVIATIONS 680-686 and 750-751). max_depth=None is
# cuML's auto, ceil(log2(max_samples_)) computed in integers rather than
# through libm log2 (DEVIATION 684). warm_start raises and sample_weight is not a fit
# parameter (UnsupportedOnGPU in cuML too) (isolation_forest.pyx:592-595). estimators_ is
# absent: the forest is flat device arrays (DEVIATION 685). as_treelite() /
# as_nvforest() are not implemented (isolation_forest/NOT_IMPLEMENTED.tsv).
# Non-finite X is refused inside the Mojo entry (DEVIATION 680); cuML accepts
# NaN and splits on it.
# DEVIATION 874: the CPython boundary retains no device pointer
# (`bindings/_mojolearn_estimators.mojo`), so a binding without the resident
# forest export keeps the training matrix and rebuilds the forest on every
# `score_samples`, `decision_function` and `predict`; the forest is a pure
# function of (random_state, tree_id, X bits), so the rebuild is bit-identical,
# at the cost of a full fit per scoring call. `fit` runs a fit plus a one-row
# scoring pass to read `offset_` and `max_samples_` back.
# DEVIATION 875: random_state=None is seed 0, not a fresh draw.
# `predict` thresholds on the device with cuML's `score > threshold ? 1 : -1`
# at threshold = -offset_, then negates; it is not re-derived from
# `decision_function`.
class IsolationForest(NumericModeMixin):
    """Isolation Forest anomaly detector with scikit-learn's interface.

    Builds random isolation trees on subsamples of `X` and scores each row by
    how quickly the trees isolate it. Parameters and defaults follow
    scikit-learn and cuML. Parameters that are not supported raise
    `NotImplementedError` instead of being ignored.

    `X` is converted to float32; NaN or infinity in `X` raises `ValueError`.
    `random_state=None` means seed 0, so the default fit is deterministic,
    unlike scikit-learn and cuML. The fitted model keeps the training matrix;
    depending on the installed build, a scoring call may rebuild the forest
    from it, which gives the same forest bit for bit.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees; at least 1.
    max_samples : 'auto', int or float, default 'auto'
        Rows per tree: 'auto' means min(256, n_samples), a positive int is
        clipped to n_samples, a float in (0, 1] is a fraction of n_samples.
    max_depth : int or None, default None
        Maximum tree depth. None means ceil(log2(``max_samples_``)).
    max_features : int or float, default 1.0
        Features per tree: an int in [1, n_features] or a float in (0, 1].
    bootstrap : bool, default False
        Draw each tree's subsample with replacement.
    contamination : 'auto' or float, default 'auto'
        Expected fraction of anomalies. 'auto' sets `offset_` to -0.5; a
        float in (0, 0.5] sets it from the training scores.
    random_state : int or None, default None
        Seed in [0, 2**32 - 1]. None means 0.
    warm_start : bool, default False
        Only False is accepted.
    n_jobs : None, default None
        Only None is accepted.
    verbose : bool, default False
        Only False is accepted.
    output_type : None, default None
        Only None is accepted; results are `mojolearn.Array`.

    Attributes
    ----------
    offset_ : float
        Shift used by `decision_function`: -0.5 for `contamination='auto'`,
        otherwise the `100 * contamination` percentile of the training
        scores (linear interpolation).
    ``max_samples_`` : int
        The resolved per-tree subsample size.
    n_features_in_ : int
        Number of features seen in `fit`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.IsolationForest(n_estimators=100).fit(X)
    >>> labels = model.predict(X)  # -1 anomaly, 1 inlier
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "outlier_detector"

    #: The binding `_run` asks through `NumericModeMixin._bind`, which is
    #: `_svm_impl._extension`'s own `_backend.binding` call; a host subclass
    #: answers the svm host binding here.
    _BINDING = "_mojolearn_svm"

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        max_depth=None,
        max_features=1.0,
        bootstrap=False,
        contamination="auto",
        random_state=None,
        warm_start=False,
        n_jobs=None,
        verbose=False,
        output_type=None,
    ):
        # scikit-learn's convention: the constructor stores its parameters
        # verbatim; `_validate_params` (run by `fit` and `load`) checks them.
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.contamination = contamination
        self.random_state = random_state
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.output_type = output_type

    def __getstate__(self):
        # The token names a forest in THIS process's binding; a copy or an
        # unpickled estimator fits its own on its first scoring call.
        state = dict(self.__dict__)
        state.pop("_if_token", None)
        state.pop("_if_lifetime", None)
        return state

    # sk-edges: no `sample_weight` parameter. The forest builder
    # (isolation_forest/) draws its subsamples unweighted, as cuML's does
    # (UnsupportedOnGPU, isolation_forest.pyx:592-595); an advertised
    # parameter could only refuse.
    def fit(self, X, y=None):
        """Fit the isolation forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : ignored

        Returns
        -------
        self
        """
        return _TREES(self, "trees_iforest_fit", (X, y), _TREES_CONTEXT)

    def score_samples(self, X):
        """Anomaly score of each row; lower is more anomalous.

        This is the negated isolation-forest anomaly score, as in
        scikit-learn and cuML.

        Returns
        -------
        Array of shape (n_samples,), float64
        """
        return _TREES(self, "trees_iforest_score_samples", (X,), _TREES_CONTEXT)

    def decision_function(self, X):
        """`score_samples(X) - offset_`, computed in float32 and returned as float64; negative means
        predicted anomalous.

        Returns
        -------
        Array of shape (n_samples,), float64
        """
        return _TREES(self, "trees_iforest_decision_function", (X,), _TREES_CONTEXT)

    def predict(self, X):
        """Label each row: -1 for an anomaly, 1 for an inlier.

        Returns
        -------
        Array of shape (n_samples,), int32
        """
        return _TREES(self, "trees_iforest_predict", (X,), _TREES_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit on `X` and return `predict(X)`."""
        return _TREES(self, "trees_iforest_fit_predict", (X, y), _TREES_CONTEXT)

    def save(self, path):
        """Write the fitted model to `path` as an npz file.

        The file holds the training matrix and the resolved parameters the
        forest is built from, so it is as large as the training data.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`.

        The saved parameters pass through the constructor again, so the same
        checks apply to a loaded file.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# `save` and `load` are one native call each (bindings/archive_schema_native.mojo
# writes this schema, the members, dtypes and order of the 0.8.55 Python
# `save`; bindings/host_archive_native.mojo `host_archive_public_iforest`
# builds the loaded model with the Python loader's refusals, the saved knobs
# checked again by the native `_validate_params`).
_native_archive_bridge.register(IsolationForest, _IFOREST_FORMAT, (
    ("need", "_x", "RuntimeError", "this estimator is not fitted yet"),
    ("iforest_knobs",),
    ("format", "format", _IFOREST_FORMAT),
    ("estimator", "estimator"),
    ("mode", "numeric_mode", "mojolearn: "),
    ("array", "x", "_x"),
    ("i8", "ints", (("int", "_n_estimators"), ("int", "_max_samples_mode"), ("int", "_max_samples_int"),
                    ("int", "_max_depth"), ("int", "_max_features_mode"), ("int", "_max_features_int"),
                    ("bool", "_bootstrap"), ("int", "_seed"), ("bool", "_contamination_auto"),
                    ("int", "n_features_in_"), ("int", "max_samples_"))),
    ("f8", "reals", (("float", "_max_samples_frac"), ("float", "_max_features_frac"),
                     ("float", "_contamination"), ("float", "offset_"))),
), trees_integral_type=numbers.Integral, trees_missing=object())
_native_trees_bridge.register(IsolationForest, "iforest")
