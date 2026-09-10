# SPDX-License-Identifier: Apache-2.0
"""Bounded sklearn protocol for RF/ET; sklearn remains an optional dependency.

Constructor parameters retain their original objects. Native configuration is
validated separately. Changing parameters clears fitted state; inference-only
legacy forest archives cannot reconstruct training parameters for cloning.
"""
import functools
import inspect
import weakref

import numpy as np
from ._arrays import _addr, _addr_ro


_FOREST_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


class _ResidentForest:
    """Own a device snapshot independently of estimator lifetime or pickling."""
    def __init__(self, native, arrays, dimensions, mode):
        self.native = native
        self.arrays = arrays
        self.dimensions = dimensions
        self.mode = mode
        self.handle = native.forest_prepare_gpu(
            *(_addr_ro(a) for a in arrays), list(dimensions))
        self._finalizer = weakref.finalize(self, native.forest_release_gpu, self.handle)

    def matches(self, native, arrays, dimensions, mode):
        return (native is self.native and dimensions == self.dimensions and mode == self.mode
                and all(a is b for a, b in zip(arrays, self.arrays)))


def forest_estimator(kind):
    """Register the explicit public constructor, including the mode wrapper."""
    def decorate(cls):
        original = cls.__init__
        signature = inspect.signature(original)
        parameters = list(signature.parameters.values())
        parameters.append(inspect.Parameter("numeric_mode", inspect.Parameter.KEYWORD_ONLY,
                                            default=None))
        signature = signature.replace(parameters=parameters)

        @functools.wraps(original)
        def initialize(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            mode = bound.arguments["numeric_mode"]
            if mode is not None and (not isinstance(mode, str) or
                    mode.strip().lower() not in ("fast", "deterministic", "identical")):
                raise ValueError("numeric_mode must be fast, deterministic, identical or None")
            self._validate_inference_engine(bound.arguments.get("inference_engine", "sequential"))
            original(self, *args, **kwargs)
            for name in cls._parameter_names:
                setattr(self, name, bound.arguments[name])
            self._has_constructor_parameters = True

        initialize.__signature__ = signature
        cls.__init__ = initialize
        cls._parameter_names = tuple(p.name for p in parameters if p.name != "self")
        cls._estimator_type = kind
        return cls
    return decorate


class ForestProtocol:
    def get_params(self, deep=True):
        for owner in type(self).__mro__:
            if "__init__" in owner.__dict__:
                if "_parameter_names" not in owner.__dict__:
                    raise TypeError("Forest subclasses with a custom constructor must register "
                                    "their parameters before cloning or refitting")
                break
        if not getattr(self, "_has_constructor_parameters", False):
            raise ValueError("This inference-only forest archive has no constructor "
                             "parameters; construct a new estimator for cloning/refitting")
        return {name: getattr(self, name) for name in self._parameter_names}

    def set_params(self, **params):
        if not params:
            return self
        values = self.get_params()
        unknown = sorted(set(params) - values.keys())
        if unknown:
            raise ValueError(f"Invalid parameter(s) {unknown} for {type(self).__name__}")
        values.update(params)
        # Apply constructor checks before dropping fitted state. Bounds checked
        # only by native fit remain deferred, as in the original constructors.
        replacement = type(self)(**values)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def _refresh_config(self):
        replacement = type(self)(**self.get_params())
        # Once a new fit begins, old nodes/classes must not survive a failure.
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)

    @staticmethod
    def _validated_mode(mode):
        if mode is None:
            return None
        if not isinstance(mode, str) or mode.strip().lower() not in (
                "fast", "deterministic", "identical"):
            raise ValueError("numeric_mode must be fast, deterministic, identical or None")
        return mode.strip().lower()

    def _capture_fit_mode(self):
        from . import _backend
        requested = self._validated_mode(getattr(self, "numeric_mode", None))
        self._fit_numeric_mode = self._validated_mode(
            _backend.default_mode() if requested is None else requested)

    def _effective_mode(self):
        from . import _backend
        requested = self._validated_mode(getattr(self, "numeric_mode", None))
        captured = getattr(self, "_fit_numeric_mode", None)
        if captured is not None:
            if requested is not None and requested != captured:
                raise ValueError(
                    f"This forest was fitted with numeric_mode={captured!r}; "
                    "refit or use set_params before changing numeric_mode")
            return captured
        # Legacy inference archives have no captured mode. Honor any stored
        # numeric_mode attribute, otherwise preserve their process default.
        return requested if requested is not None else _backend.default_mode()

    def _bind(self, name=None):
        from . import _backend
        return _backend.binding(name or self._BINDING, self._effective_mode())

    @staticmethod
    def _validate_inference_engine(engine):
        if not isinstance(engine, str) or engine not in ("sequential", "parallel_groves"):
            raise ValueError("inference_engine must be 'sequential' or 'parallel_groves'")
        return engine

    def _prediction_engine(self):
        return self._validate_inference_engine(getattr(self, "inference_engine", "sequential"))

    def _prediction_function(self, sequential_name):
        engine = self._prediction_engine()
        native = self._bind()
        name = sequential_name if engine == "sequential" else sequential_name + "_gpu_parallel"
        function = getattr(native, name, None)
        if function is None:
            raise RuntimeError("rebuild the forest binding for inference_engine=" + repr(engine))
        return function

    def _resident_prediction_function(self, native):
        return native.forest_predict_resident_into_gpu

    def _predict_forest(self, sequential_name, X, out):
        """Shared RF/ET dispatch; cache nvForest-style owned device model state.

        cuML 26.08.00 randomforest_common.pyx:675-693 caches its nvForest
        model; :350-353 omits that device object from pickle. Our flat model
        uses immutable host snapshots to make invalidation unambiguous.
        """
        engine = self._prediction_engine()
        native = self._bind()
        rows, features = X.shape
        dimensions = (int(features), int(self._n_trees), int(self._num_outputs))
        arrays = tuple(getattr(self, name) for name in _FOREST_ARRAYS)
        if engine == "sequential":
            return self._prediction_function(sequential_name)(
                *(_addr_ro(a) for a in arrays), _addr_ro(X),
                _addr(out), [int(rows), *dimensions])
        required = ("forest_prepare_gpu", "forest_predict_resident_into_gpu", "forest_release_gpu")
        if any(not callable(getattr(native, name, None)) for name in required):
            raise RuntimeError("rebuild the forest binding for resident parallel_groves inference")
        mode = self._effective_mode()
        resident = getattr(self, "_resident_forest", None)
        if resident is None or not resident.matches(native, arrays, dimensions, mode):
            # A bytes owner cannot be made writable again via setflags(). A
            # caller retaining an old mutable private-array alias cannot alter
            # the device snapshot or the host model used by save/sequential.
            dtypes = (np.int32, np.int32, np.float32, np.int32, np.float32)
            for a, dtype in zip(arrays, dtypes):
                if not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != 1:
                    raise ValueError("forest model arrays must have their original flat dtypes")
            nodes = arrays[1].size
            if (arrays[0].size != dimensions[1] + 1 or nodes < 1
                    or arrays[2].size != nodes or arrays[3].size != nodes
                    or arrays[4].size != nodes * dimensions[2]
                    or int(arrays[0][-1]) != nodes):
                raise ValueError("forest model array shapes do not match metadata")
            frozen = tuple(np.frombuffer(a.tobytes(), dtype=a.dtype) for a in arrays)
            resident = _ResidentForest(native, frozen, dimensions, mode)
            for name, a in zip(_FOREST_ARRAYS, frozen):
                setattr(self, name, a)
            self._resident_forest = resident
        return self._resident_prediction_function(native)(
            resident.handle, _addr_ro(X), _addr(out),
            [int(rows), int(features), dimensions[2]])

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_resident_forest", None)
        return state

    def _archive_inference_metadata(self, arrays, sequential_format):
        if self._prediction_engine() == "parallel_groves":
            arrays["format"] = np.asarray(sequential_format + "-parallel-groves-1")
            arrays["numeric_mode"] = np.asarray(self._effective_mode())

    def _restore_inference_metadata(self, arrays, sequential_format):
        from . import _serialize
        if _serialize.scalar_str(arrays, "format") == sequential_format:
            self.inference_engine = "sequential"
        else:
            self.inference_engine = "parallel_groves"
            mode = self._validated_mode(_serialize.scalar_str(arrays, "numeric_mode"))
            self.numeric_mode = mode
            self._fit_numeric_mode = mode

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_offsets")

    def __sklearn_tags__(self):
        from sklearn.utils import Tags, TargetTags, ClassifierTags, RegressorTags
        classifier = self._estimator_type == "classifier"
        return Tags(estimator_type=self._estimator_type,
                    target_tags=TargetTags(required=True),
                    classifier_tags=ClassifierTags() if classifier else None,
                    regressor_tags=None if classifier else RegressorTags())

    def score(self, X, y, sample_weight=None):
        """Mode-aware GPU accuracy or Float32 R²; weighted scoring is unsupported.

        Class labels are compared on the host without narrowing their values;
        the GPU accuracy kernel receives exact integer equality indicators.
        Regression scoring uses the same Float32 target domain as tree fitting.
        """
        from . import _metrics_impl as metrics
        if sample_weight is not None:
            raise NotImplementedError("Forest score does not yet support sample_weight")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError("score requires one-dimensional targets")
        prediction = np.asarray(self.predict(X))
        if prediction.shape != target.shape:
            raise ValueError("score target and prediction lengths differ")
        mode = self._effective_mode()
        if self._estimator_type == "classifier":
            equal = np.asarray(target == prediction, dtype=np.int32)
            return metrics.accuracy_score(np.ones(equal.shape, dtype=np.int32),
                                          equal, numeric_mode=mode)
        return metrics.r2_score(np.asarray(target, dtype=np.float32),
                                np.asarray(prediction, dtype=np.float32),
                                numeric_mode=mode)
