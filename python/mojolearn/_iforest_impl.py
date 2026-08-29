"""Isolation Forest on the GPU, mirroring cuML's `IsolationForest`.

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
    from the same seed. This port takes the first draw as the high word,
    by name, and that choice has never been checked against a cuML binary.
    Until it is, agreement with cuML is a belief, not a measurement. It is
    one number off a real NVIDIA GPU to close.
  * The card stages added on 2026-08-24 (`split.bounds`, `split.choice`,
    `rng.final`) HAD NOT BEEN COMPILED when this wrapper was written.
"""

import numpy as np

from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

# THE EXTENSION LOADER LIVES IN `_svm_impl`, NOT BECAUSE THESE TWO LANES
# HAVE ANYTHING TO DO WITH EACH OTHER -- they do not -- but because both
# are compiled into ONE extension module, `_mojolearn_svm`, by
# `bindings/build_svm.sh`. One loader means one place that decides which
# numeric mode's binary is open, and one cache so the module is loaded
# once. Read `_svm_impl._extension` before changing anything here.
from ._svm_impl import _extension

_WANT_SCORE_SAMPLES = 0
_WANT_DECISION_FUNCTION = 1
_WANT_PREDICT = 2


class IsolationForest(NumericModeMixin):
    """Isolation Forest backed by the ported cuML path
    (`isolation_forest/`, DEVIATIONS 680-686 and 750-751), the
    scikit-learn surface.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter:

        n_estimators    honored   the tree count (cuML's default 100)
        max_samples     honored   'auto' (= min(256, n_samples)), a
                                  positive int (clipped to n_samples), or
                                  a float in (0, 1]; cuML's `fit`
                                  resolution, transcribed
        max_depth       honored   None (the default) is their auto, an
                                  integer ceil(log2(max_samples_)) computed
                                  in integers rather than through a libm
                                  log2 (DEVIATION 684); a positive int is
                                  taken as given
        max_features    honored   a float in (0, 1] (default 1.0) or an int
                                  in [1, n_features]
        bootstrap       honored   whether the per-tree subsample is drawn
                                  with replacement
        contamination   honored   'auto' (offset_ = -0.5) or a float in
                                  (0, 0.5], which takes the training-score
                                  quantile exactly as theirs does
        random_state    honored   an int in [0, 2**32 - 1]. None is 0 HERE,
                                  not a fresh draw -- DEVIATION 875 below
        warm_start      refused   `UnsupportedOnGPU` upstream too
                                  (isolation_forest.pyx:592-595)
        sample_weight   refused   in fit(); `UnsupportedOnGPU` upstream too
        n_jobs          refused   scikit-learn's thread count; there is no
                                  CPU path in this library to spread
        verbose         refused   anything truthy; this port prints no log
                                  lines, so accepting it would be
                                  accepting-and-ignoring
        output_type     refused   a cuML-internal array-type selector; this
                                  package returns NumPy
        estimators_     absent    the per-tree Python objects are
                                  scikit-learn's; the forest here is four
                                  flat device arrays (DEVIATION 685)
        as_treelite(),  absent    the Treelite / nvForest export is not
        as_nvforest()             ported (isolation_forest/UNPORTED.tsv)

    Non-finite cells of `X` are refused by name inside the Mojo entry
    (DEVIATION 680), with the row and column, rather than being split on.
    Theirs accepts a NaN and splits on it.

    DEVIATION 874: EVERY SCORING CALL REFITS THE FOREST. The fitted forest
    is eight device buffers and the CPython boundary retains no device
    pointer (`bindings/_mojolearn_estimators.mojo`'s header states that
    rule for the whole package), so `fit` cannot hand a live model to
    `predict`. This class keeps the training matrix instead and rebuilds
    the forest on every `score_samples`, `decision_function` and
    `predict`. It is CORRECT -- the forest is a pure function of
    `(random_state, tree_id, X bits)`, which is the property the whole
    lane is built around, so the rebuild is the same forest bit for bit --
    and it COSTS a full fit per scoring call. `fit` itself runs a fit plus
    a one-row scoring pass, which is the cheapest way to read `offset_`
    and `max_samples_` back out.

    DEVIATION 875: `random_state=None` IS SEED 0 HERE, NOT A FRESH DRAW.
    scikit-learn and cuML both take a fresh entropy source for None. A fit
    whose seed nobody recorded cannot be reproduced, and a reproducible
    fit is this library's product, so None is 0 and it is said out loud
    rather than left for a caller to discover from two runs agreeing.

    Attributes
    ----------
    offset_ : float
        The decision-function shift. -0.5 for `contamination='auto'`,
        otherwise the `100 * contamination` percentile of the training
        scores under numpy's default `linear` interpolation.
    max_samples_ : int
        The resolved per-tree subsample size.
    n_features_in_ : int

    Methods
    -------
    score_samples(X)
        The negated anomaly score, cuML's `-paper_score`. LOWER is more
        anomalous, which is scikit-learn's convention and theirs.
    decision_function(X)
        `score_samples(X) - offset_`, in float32 as the Python layer
        upstream computes it. Negative is predicted anomalous.
    predict(X)
        -1 for an anomaly, 1 for an inlier, thresholded ON THE DEVICE by
        cuML's own `score > threshold ? 1 : -1` at `threshold = -offset_`
        and then negated, not re-derived from `decision_function`.
    """

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
        n_estimators = int(n_estimators)
        if n_estimators < 1:
            raise ValueError(
                f"mojolearn IsolationForest: n_estimators must be at least 1, "
                f"got {n_estimators!r}"
            )
        # max_samples: 0 = 'auto', 1 = int, 2 = float fraction. cuML's `fit`
        # (isolation_forest.pyx:663-702) does the resolving; the mode is how
        # this surface tells an int from a float across the boundary, since
        # a params list carries no types.
        if isinstance(max_samples, str):
            if max_samples.lower() != "auto":
                raise ValueError(
                    f"mojolearn IsolationForest: max_samples={max_samples!r} "
                    "is not a name; it is 'auto', a positive int, or a float "
                    "in (0.0, 1.0]"
                )
            self._max_samples_mode = 0
            self._max_samples_int = 256
            self._max_samples_frac = 1.0
        elif isinstance(max_samples, (bool, np.bool_)):
            raise ValueError(
                "mojolearn IsolationForest: max_samples must be 'auto', an "
                "int, or a float, not a bool"
            )
        elif isinstance(max_samples, (int, np.integer)):
            if int(max_samples) <= 0:
                raise ValueError(
                    "mojolearn IsolationForest: max_samples must be a "
                    f"positive integer, got {max_samples!r}"
                )
            self._max_samples_mode = 1
            self._max_samples_int = int(max_samples)
            self._max_samples_frac = 1.0
        else:
            v = float(max_samples)
            if not (0.0 < v <= 1.0):
                raise ValueError(
                    "mojolearn IsolationForest: float max_samples must be in "
                    f"(0.0, 1.0], got {max_samples!r}"
                )
            self._max_samples_mode = 2
            self._max_samples_int = 256
            self._max_samples_frac = v

        if max_depth is None:
            self._max_depth = -1
        else:
            self._max_depth = int(max_depth)
            if self._max_depth < 1:
                raise ValueError(
                    "mojolearn IsolationForest: max_depth is None (their "
                    f"auto) or a positive int, got {max_depth!r}"
                )

        # max_features: 0 = float fraction (their default 1.0), 1 = int.
        if isinstance(max_features, (bool, np.bool_)):
            raise ValueError(
                "mojolearn IsolationForest: max_features must be an int or a "
                "float, not a bool"
            )
        if isinstance(max_features, (int, np.integer)):
            if int(max_features) < 1:
                raise ValueError(
                    "mojolearn IsolationForest: max_features must be an int "
                    "in [1, n_features] or a float in (0.0, 1.0], got "
                    f"{max_features!r}"
                )
            self._max_features_mode = 1
            self._max_features_int = int(max_features)
            self._max_features_frac = 1.0
        else:
            v = float(max_features)
            if not (0.0 < v <= 1.0):
                raise ValueError(
                    "mojolearn IsolationForest: max_features must be an int "
                    "in [1, n_features] or a float in (0.0, 1.0], got "
                    f"{max_features!r}"
                )
            self._max_features_mode = 0
            self._max_features_int = 0
            self._max_features_frac = v

        if isinstance(contamination, str):
            if contamination.lower() != "auto":
                raise ValueError(
                    "mojolearn IsolationForest: contamination must be 'auto' "
                    f"or a float in the range (0.0, 0.5], got "
                    f"{contamination!r}"
                )
            self._contamination_auto = True
            self._contamination = 0.0
        else:
            v = float(contamination)
            if not (0.0 < v <= 0.5):
                raise ValueError(
                    "mojolearn IsolationForest: contamination must be 'auto' "
                    f"or a float in the range (0.0, 0.5], got "
                    f"{contamination!r}"
                )
            self._contamination_auto = False
            self._contamination = v

        if random_state is None:
            seed = 0
        else:
            seed = int(random_state)
            if not (0 <= seed <= 4294967295):
                raise ValueError(
                    "mojolearn IsolationForest: expected "
                    f"0 <= random_state <= 2**32 - 1, got {random_state!r}"
                )
        if warm_start:
            raise NotImplementedError(
                "mojolearn IsolationForest: warm_start=True is not supported "
                "(cuML raises UnsupportedOnGPU for it too)"
            )
        if n_jobs is not None:
            raise NotImplementedError(
                "mojolearn IsolationForest: n_jobs is refused; it is "
                "scikit-learn's CPU thread count and there is no CPU path in "
                "this library to spread"
            )
        if verbose:
            raise NotImplementedError(
                "mojolearn IsolationForest: verbose is refused; this port "
                "prints no log lines, so accepting it would be accepting-"
                "and-ignoring"
            )
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn IsolationForest: output_type is a cuML-internal "
                "array-type selector; this package returns NumPy"
            )

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.contamination = contamination
        self.random_state = random_state
        self._seed = seed
        self.warm_start = False
        self.n_jobs = None
        self.verbose = False
        self.output_type = None

    def _params(self, n_train, n_features, n_query, want):
        # ORDER MATCHES bindings/_mojolearn_svm.mojo::iforest_run_binding.
        # n_train, n_features, n_query, n_estimators, max_samples_mode,
        # max_samples_int, max_samples_frac, max_depth, max_features_mode,
        # max_features_int, max_features_frac, bootstrap, random_state,
        # contamination_auto, contamination, want
        return [
            n_train,
            n_features,
            n_query,
            self.n_estimators,
            self._max_samples_mode,
            self._max_samples_int,
            self._max_samples_frac,
            self._max_depth,
            self._max_features_mode,
            self._max_features_int,
            self._max_features_frac,
            1 if self.bootstrap else 0,
            self._seed,
            1 if self._contamination_auto else 0,
            self._contamination,
            want,
        ]

    def _run(self, X, want):
        """One fit-and-score. DEVIATION 874: the fit happens here, every
        time, on the training matrix `fit` kept."""
        if not hasattr(self, "_x"):
            raise ValueError("mojolearn IsolationForest: call fit() first")
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn IsolationForest: X has {q.shape[1]} features, fit "
                f"saw {self.n_features_in_}"
            )
        n_query = q.shape[0]
        train = self._x  # kept in a local: the Mojo side borrows the address
        values = np.empty(n_query, dtype=np.float32)
        labels = np.empty(n_query, dtype=np.int32)
        info = np.empty(3, dtype=np.float64)
        _extension(getattr(self, 'numeric_mode', None)).iforest_run(
            _addr_ro(train),
            _addr_ro(q),
            _addr(values),
            _addr(labels),
            _addr(info),
            self._params(train.shape[0], train.shape[1], n_query, want),
        )
        self.offset_ = float(info[0])
        self.max_samples_ = int(info[1])
        return labels if want == _WANT_PREDICT else values

    def fit(self, X, y=None, sample_weight=None):
        """Fits, and reads back `offset_`, `max_samples_` and
        `n_features_in_`. The forest itself is NOT kept: it is rebuilt on
        every scoring call (DEVIATION 874), so this runs the fit plus a
        one-row scoring pass, which is the cheapest call the entry point
        accepts."""
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn IsolationForest: sample_weight is not supported "
                "(cuML raises UnsupportedOnGPU for it too)"
            )
        x, self.input_copied_ = as_f32_c(X, "X")
        self._x = x  # kept alive; every scoring call refits from it
        self.n_features_in_ = x.shape[1]
        self._run(x[:1], _WANT_SCORE_SAMPLES)
        return self

    def score_samples(self, X):
        return self._run(X, _WANT_SCORE_SAMPLES)

    def decision_function(self, X):
        return self._run(X, _WANT_DECISION_FUNCTION)

    def predict(self, X):
        return self._run(X, _WANT_PREDICT)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y=y, sample_weight=sample_weight).predict(X)
