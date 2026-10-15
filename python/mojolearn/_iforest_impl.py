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
import weakref

from . import _serialize
from ._array import Array
from ._buffer import addr, addr_ro, all_finite, as_f32_c, empty
from ._labels import is_bool
from ._mode import NumericModeMixin

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


# Maintainer notes (moved from the docstring): the GPU implementation is
# `isolation_forest/` (DEVIATIONS 680-686 and 750-751). max_depth=None is
# cuML's auto, ceil(log2(max_samples_)) computed in integers rather than
# through libm log2 (DEVIATION 684). warm_start and sample_weight raise
# UnsupportedOnGPU in cuML too (isolation_forest.pyx:592-595). estimators_ is
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
        elif is_bool(max_samples):
            raise ValueError(
                "mojolearn IsolationForest: max_samples must be 'auto', an "
                "int, or a float, not a bool"
            )
        elif isinstance(max_samples, numbers.Integral):
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
        if is_bool(max_features):
            raise ValueError(
                "mojolearn IsolationForest: max_features must be an int or a "
                "float, not a bool"
            )
        if isinstance(max_features, numbers.Integral):
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
                "mojolearn IsolationForest: verbose is refused; this implementation "
                "prints no log lines, so accepting it would be accepting-"
                "and-ignoring"
            )
        if output_type is not None:
            raise NotImplementedError(
                "mojolearn IsolationForest: output_type is a cuML-internal "
                "array-type selector; this package returns mojolearn Arrays"
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

    def _run(self, X, want, fresh=False):
        """One fit-and-score. DEVIATION 874: the fit happens here, every
        time, on the training matrix `fit` kept, unless the binary keeps the
        forest resident (IDN_IF_RESIDENT); `fresh` (what `fit` passes) asks
        such a binary for a new forest."""
        if not hasattr(self, "_x"):
            raise ValueError("mojolearn IsolationForest: call fit() first")
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn IsolationForest: X has {q.shape[1]} features, fit "
                f"saw {self.n_features_in_}"
            )
        n_query = q.shape[0]
        train = self._x  # kept in a local: the Mojo side borrows the address
        # DEVIATION 2344: `_buffer.empty` output `Array`s; nothing here
        # reads them on the host except the three-slot `info`.
        # The native contract touches exactly one result address according
        # to `want`; do not allocate the other million-row boundary buffer.
        values = None if want == _WANT_PREDICT else empty((n_query,), "<f4")
        labels = empty((n_query,), "<i4") if want == _WANT_PREDICT else None
        b = self._bind("_mojolearn_svm")
        # Lane fam2-forests (IDN_IF_RESIDENT): a binary that keeps the fitted
        # forest resident takes a 17th param, the token of this estimator's
        # forest (0 = fit a fresh one), and returns the token in info[3].
        # A scoring call then runs no fit. Other binaries refit per call
        # (DEVIATION 874), unchanged.
        probe = getattr(b, "iforest_resident", None)
        resident = probe is not None and int(probe()) == 1
        info = empty((4 if resident else 3,), "<f8")
        params = self._params(train.shape[0], train.shape[1], n_query, want)
        if resident:
            params.append(0 if fresh else int(self.__dict__.get("_if_token", 0)))
        b.iforest_run(
            addr_ro(train, name="X (training)"),
            addr_ro(q, name="X"),
            0 if values is None else addr(values, name="scores"),
            0 if labels is None else addr(labels, name="labels"),
            addr(info, name="info"),
            params,
        )
        self.offset_ = float(info[0])
        self.max_samples_ = int(info[1])
        if resident:
            token = int(info[3])
            if token != self.__dict__.get("_if_token", 0):
                self._release_resident()
                # Run while Python and the GPU binding are still alive, even
                # for estimators retained until interpreter shutdown. Keep the
                # native entry itself, not a late module lookup in __del__.
                self._if_finalizer = weakref.finalize(
                    self, b.iforest_resident_release, token
                )
            self._if_token = token
        return labels if want == _WANT_PREDICT else values

    def _release_resident(self):
        """Drop this estimator's resident forest in the binding, if any."""
        token = self.__dict__.pop("_if_token", 0)
        finalizer = self.__dict__.pop("_if_finalizer", None)
        if finalizer is not None:
            try:
                finalizer()
            except Exception:  # noqa: BLE001 - match explicit release semantics
                pass
            return
        if not token:
            return
        try:
            release = getattr(self._bind("_mojolearn_svm"), "iforest_resident_release", None)
            if release is not None:
                release(int(token))
        except Exception:  # noqa: BLE001 - a release that cannot run frees nothing
            pass

    def __del__(self):
        self._release_resident()

    def __getstate__(self):
        # The token names a forest in THIS process's binding; a copy or an
        # unpickled estimator fits its own on its first scoring call.
        state = dict(self.__dict__)
        state.pop("_if_token", None)
        state.pop("_if_finalizer", None)
        return state

    def fit(self, X, y=None, sample_weight=None):
        """Fit the isolation forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : ignored
        sample_weight : None, default None
            Only None is accepted; other values raise `NotImplementedError`.

        Returns
        -------
        self
        """
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn IsolationForest: sample_weight is not supported "
                "(cuML raises UnsupportedOnGPU for it too)"
            )
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        # A binary that scans the uploaded matrix on the device refuses a
        # non-finite X inside the fit, so the host scan here would be the same
        # refusal twice: `iforest_device_scan` (lane gap-trees-nv,
        # IF_DEVICE_TRANSPOSE) or `iforest_device_finite_scan` (lane
        # apple-fast-trees2, IF_FAST_ROWMAJOR, FAST on Apple).
        b = self._bind("_mojolearn_svm")
        scans = [getattr(b, n, None) for n in ("iforest_device_scan", "iforest_device_finite_scan")]  # glue: looks up two binding entries
        device_scan = any(f is not None and int(f()) == 1 for f in scans)  # glue: checks two binding entries
        if not device_scan and not all_finite(x):
            # was a bare Exception from the native fit; scikit-learn raises
            # ValueError for the same input
            raise ValueError("mojolearn IsolationForest: X contains NaN or infinity")
        had = {k: self.__dict__[k] for k in ("_x", "n_features_in_") if k in self.__dict__}  # glue: saves two attribute names
        self._release_resident()  # the previous fit's resident forest, if any
        self._x = x  # retained for bindings that reconstruct a missing forest
        self.n_features_in_ = x.shape[1]
        try:
            self._run(x[:1], _WANT_SCORE_SAMPLES, fresh=True)  # one row, an `Array` copy
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if device_scan and ("Input X contains" in msg or "does not accept non-finite" in msg):
                for k in ("_x", "n_features_in_"):  # glue: restores two attribute names
                    self.__dict__.pop(k, None)
                self.__dict__.update(had)
                raise ValueError(
                    "mojolearn IsolationForest: X contains NaN or infinity"
                ) from None
            raise
        return self

    def score_samples(self, X):
        """Anomaly score of each row; lower is more anomalous.

        This is the negated isolation-forest anomaly score, as in
        scikit-learn and cuML.

        Returns
        -------
        Array of shape (n_samples,), float32
        """
        return self._run(X, _WANT_SCORE_SAMPLES)

    def decision_function(self, X):
        """`score_samples(X) - offset_`, computed in float32; negative means
        predicted anomalous.

        Returns
        -------
        Array of shape (n_samples,), float32
        """
        return self._run(X, _WANT_DECISION_FUNCTION)

    def predict(self, X):
        """Label each row: -1 for an anomaly, 1 for an inlier.

        Returns
        -------
        Array of shape (n_samples,), int32
        """
        return self._run(X, _WANT_PREDICT)

    def fit_predict(self, X, y=None, sample_weight=None):
        """Fit on `X` and return `predict(X)`."""
        return self.fit(X, y=y, sample_weight=sample_weight).predict(X)

    def save(self, path):
        """Write the fitted model to `path` as an npz file.

        The file holds the training matrix and the resolved parameters the
        forest is built from, so it is as large as the training data.
        """
        if not hasattr(self, "_x"):
            raise RuntimeError("this estimator is not fitted yet")
        from .linear_model import _saved_mode
        return _serialize.write_npz(path, {
            "format": _IFOREST_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "x": self._x,
            "ints": Array.from_list([
                int(self.n_estimators), self._max_samples_mode, self._max_samples_int,
                self._max_depth, self._max_features_mode, self._max_features_int,
                1 if self.bootstrap else 0, self._seed, 1 if self._contamination_auto else 0,
                int(self.n_features_in_), int(self.max_samples_),
            ], "<i8"),
            "reals": Array.from_list([
                float(self._max_samples_frac), float(self._max_features_frac),
                float(self._contamination), float(self.offset_),
            ], "<f8"),
        })

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`.

        The saved parameters pass through the constructor again, so the same
        checks apply to a loaded file.
        """
        from .linear_model import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _IFOREST_FORMAT)
        _check_saved_by(arrays, path, cls)
        ints = _serialize.exact(arrays, "ints", "<i8")
        reals = _serialize.exact(arrays, "reals", "<f8")
        if ints.size != _IFOREST_INTS or reals.size != _IFOREST_REALS:
            raise ValueError(
                f"mojolearn: {path!r} holds {ints.size} ints and {reals.size} reals, "
                f"{_IFOREST_INTS} and {_IFOREST_REALS} are needed"
            )
        (n_est, ms_mode, ms_int, depth, mf_mode, mf_int, boot, seed, c_auto, nf,
         ms_resolved) = (int(ints[i]) for i in range(_IFOREST_INTS))  # glue: unpacks the fixed int vector
        ms_frac, mf_frac, contamination, offset = (float(reals[i]) for i in range(_IFOREST_REALS))  # glue: unpacks the fixed real vector
        if ms_mode not in (0, 1, 2) or mf_mode not in (0, 1) or c_auto not in (0, 1) or boot not in (0, 1):
            raise ValueError(f"mojolearn: {path!r} carries an unknown knob mode")
        obj = cls(
            n_estimators=n_est,
            max_samples="auto" if ms_mode == 0 else (ms_int if ms_mode == 1 else ms_frac),
            max_depth=None if depth == -1 else depth,
            max_features=mf_int if mf_mode == 1 else mf_frac,
            bootstrap=bool(boot),
            contamination="auto" if c_auto else contamination,
            random_state=seed,
        )
        _restore_mode(obj, arrays)
        x = _serialize.exact(arrays, "x", "<f4")
        if x.ndim != 2 or x.shape[1] != nf or x.shape[0] < 1:
            raise ValueError(f"mojolearn: {path!r} x has shape {tuple(x.shape)}, meta says {nf} features")
        obj._x = x
        obj.n_features_in_ = nf
        obj.offset_ = offset
        obj.max_samples_ = ms_resolved
        return obj
