# SPDX-License-Identifier: Apache-2.0
"""Bounded sklearn protocol for RF/ET; sklearn remains an optional dependency.

Constructor parameters retain their original objects. Native configuration is
validated separately. Changing parameters clears fitted state; inference-only
legacy forest archives cannot reconstruct training parameters for cloning.
"""
import functools
import inspect
import weakref

from ._array import Array
from ._buffer import _materialize, full, as_f32_c
from ._labels import flatten_labels
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
        # sklearn owns this transient context while Pipeline.fit is active.
        context = self.__dict__.get("_parent_callback_ctx")
        # Once a new fit begins, old nodes/classes must not survive a failure.
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        if context is not None:
            self._parent_callback_ctx = context

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
        # DEVIATION 2483: this export is predict_into[REUSE_IO=True], so X and
        # output already cross as borrowed pointers without List staging.
        # Keep device-buffer reuse; plain into would allocate on every call.
        return native.forest_predict_resident_reuse_gpu

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
        required = ("forest_prepare_gpu", "forest_predict_resident_reuse_gpu", "forest_release_gpu")
        if any(not callable(getattr(native, name, None)) for name in required):
            raise RuntimeError("rebuild the forest binding for resident parallel_groves inference")
        mode = self._effective_mode()
        resident = getattr(self, "_resident_forest", None)
        if resident is None or not resident.matches(native, arrays, dimensions, mode):
            # A bytes owner cannot be made writable again via setflags(). A
            # caller retaining an old mutable private-array alias cannot alter
            # the device snapshot or the host model used by save/sequential.
            dtypes = ("<i4", "<i4", "<f4", "<i4", "<f4")
            arrays = tuple(_materialize(a, "forest model")[0] for a in arrays)
            for a, dtype in zip(arrays, dtypes):
                if not isinstance(a, Array) or a.dtype != dtype or a.ndim != 1:
                    raise ValueError("forest model arrays must have their original flat dtypes")
            nodes = arrays[1].size
            if (arrays[0].size != dimensions[1] + 1 or nodes < 1
                    or arrays[2].size != nodes or arrays[3].size != nodes
                    or arrays[4].size != nodes * dimensions[2]
                    or int(arrays[0][-1]) != nodes):
                raise ValueError("forest model array shapes do not match metadata")
            frozen = tuple(Array.from_buffer(memoryview(a.tobytes()).cast("i" if a.dtype == "<i4" else "f")) for a in arrays)
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
            arrays["format"] = sequential_format + "-parallel-groves-1"
            arrays["numeric_mode"] = self._effective_mode()

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
        if getattr(y, "ndim", 1) != 1:
            raise ValueError("score requires one-dimensional targets")
        target = flatten_labels(y)
        prediction = self.predict(X)
        if len(prediction) != len(target):
            raise ValueError("score target and prediction lengths differ")
        mode = self._effective_mode()
        if self._estimator_type == "classifier":
            equal = Array.from_list([int(a == b) for a, b in zip(target, prediction)], "<i4")
            return metrics.accuracy_score(full(equal.shape, 1, "<i4"), equal, numeric_mode=mode)
        return metrics.r2_score(as_f32_c(y, ndim=1, name="y")[0],
                                as_f32_c(prediction, ndim=1, name="prediction")[0],
                                numeric_mode=mode)


def _export_fit_result(native, descriptor, *, compare_legacy=False):
    """WP2a: copy one owned native fit into Arrays, always releasing its handle.

    Descriptor: [handle, trees, nodes, outputs, small integer metadata]. Export
    receives exact capacities as well as addresses. The optional diagnostic
    reads the SAME fitted handle through the retained List exporter before
    release; it is a correctness gate, never a second fit or production path.
    Native same-fit/save gates cover all three modes on Metal (DEVIATION 2482).
    """
    import numbers
    from ._buffer import empty

    if not isinstance(descriptor, (list, tuple)) or not descriptor:
        raise ValueError('native forest export returned an invalid descriptor')
    handle = descriptor[0]
    if isinstance(handle, bool) or not isinstance(handle, numbers.Integral) or handle < 1:
        raise ValueError('native forest export returned an invalid handle')
    try:
        if len(descriptor) != 5:
            raise ValueError('native forest export descriptor requires five fields')
        _, trees, nodes, outputs, meta = descriptor
        counts = (trees, nodes, outputs)
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in counts):
            raise ValueError('native forest export counts must be integers')
        if not (1 <= trees < 2147483647 and nodes >= trees and outputs >= 1
                and nodes <= 2147483647 // outputs):
            raise ValueError('native forest export exceeds supported counts')
        if not isinstance(meta, (list, tuple)) or len(meta) < 2 or list(meta[:2]) != [trees, outputs]:
            raise ValueError('native forest export metadata disagrees with counts')
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) for v in meta):
            raise ValueError('native forest export metadata must contain integers')
        dtypes = ('<i4', '<i4', '<f4', '<i4', '<f4')
        sizes = (trees + 1, nodes, nodes, nodes, nodes * outputs)
        arrays = tuple(empty((size,), dtype) for size, dtype in zip(sizes, dtypes))
        native.forest_export(handle, *(_addr(a) for a in arrays), list(counts))
        if compare_legacy:
            old = native.forest_export_legacy(handle)
            if len(old) != 6 or list(old[5]) != list(meta):
                raise RuntimeError('forest export legacy metadata mismatch')
            for name, actual, values, dtype in zip(_FOREST_ARRAYS, arrays, old[:5], dtypes):
                expected = Array.from_list(values, dtype)
                if actual.tobytes() != expected.tobytes():
                    raise RuntimeError('forest export byte mismatch: ' + name)
        return (*arrays, list(meta))
    finally:
        native.forest_export_release(handle)


def _forest_fit_function(native, name):
    """WP2a caller-buffer default; legacy/verify are diagnostic comparison arms."""
    import os
    selection = os.environ.get('MOJOLEARN_FOREST_EXPORT', 'into')
    if selection == 'legacy':
        return getattr(native, name)
    if selection not in ('into', 'verify'):
        raise ValueError('MOJOLEARN_FOREST_EXPORT must be legacy, into or verify')
    entry = getattr(native, name + '_export', None)
    if not callable(entry):
        raise RuntimeError('rebuild the forest binding for caller-buffer model export')
    def fit(*args):
        return _export_fit_result(native, entry(*args), compare_legacy=selection == 'verify')
    return fit


def _forest_fit_arrays(result):
    """One RF/ET model conversion; native exported arrays are retained directly."""
    *fields, meta = result
    if len(fields) != 5:
        raise ValueError('forest fit must return five model fields and metadata')
    dtypes = ('<i4', '<i4', '<f4', '<i4', '<f4')
    arrays = []
    for field, dtype in zip(fields, dtypes):
        if isinstance(field, Array):
            if field.dtype != dtype or field.ndim != 1:
                raise ValueError('forest fit exported an unexpected model dtype or shape')
            arrays.append(field)
        else:
            arrays.append(Array.from_list(field, dtype))
    return (*arrays, meta)
