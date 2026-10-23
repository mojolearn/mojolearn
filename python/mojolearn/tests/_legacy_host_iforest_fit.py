# SPDX-License-Identifier: Apache-2.0
"""Frozen 32011bb32 CPU IsolationForest fit and validation test oracle."""
import numbers
from mojolearn import _sk_validate
from mojolearn._buffer import as_f32_c, all_finite
from mojolearn._labels import is_bool
from mojolearn.tests._legacy_host_inference import iforest


def parameters(self):
        """The parameter checks and the native knobs they resolve to, run at
        fit. The parameters themselves stay as given."""
        n_estimators = self.n_estimators
        max_samples = self.max_samples
        max_depth = self.max_depth
        max_features = self.max_features
        bootstrap = self.bootstrap
        contamination = self.contamination
        random_state = self.random_state
        warm_start = self.warm_start
        n_jobs = self.n_jobs
        verbose = self.verbose
        output_type = self.output_type
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

        self._n_estimators = n_estimators
        self._bootstrap = bool(bootstrap)
        self._seed = seed

def fit(model, X, y=None):
    """The old public wrapper and CPU body, using its unchanged numerical entry."""
    state = vars(model)
    check = None
    args, kwargs = (X, y), {}
    if not state.get("_mojolearn_fitting"):
        args, kwargs, check = _sk_validate.before_fit(model, "fit", (0, 1, "y"), args, kwargs)
    state["_mojolearn_fitting"] = state.get("_mojolearn_fitting", 0) + 1
    try:
        parameters(model)
        x, model.input_copied_ = as_f32_c(args[0], ndim=2, name="X")
        binding = model._bind("_mojolearn_svm")
        scans = [getattr(binding, name, None) for name in ("iforest_device_scan", "iforest_device_finite_scan")]
        assert not any(entry is not None and int(entry()) == 1 for entry in scans), "reference requires the explicit CPU numerical entry"
        if not all_finite(x):
            raise ValueError("mojolearn IsolationForest: X contains NaN or infinity")
        # The removed IsolationForest._release_resident: drop the token and
        # its lifetime.
        vars(model).pop("_if_token", None)
        vars(model).pop("_if_lifetime", None)
        model._x = x
        model.n_features_in_ = x.shape[1]
        iforest(model, x[:1], 0, public=False)
    except Exception as error:
        if check is not None:
            _sk_validate.reraise(model, check, error)
        raise
    finally:
        depth = state.get("_mojolearn_fitting", 1) - 1
        if depth > 0:
            state["_mojolearn_fitting"] = depth
        else:
            state.pop("_mojolearn_fitting", None)
    state["_mojolearn_fitted"] = True
    if check is not None:
        _sk_validate.after_fit(model, check)
    return model
