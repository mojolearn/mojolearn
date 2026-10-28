# SPDX-License-Identifier: Apache-2.0
"""Bounded sklearn protocol for RF/ET; sklearn remains an optional dependency.

Constructor parameters retain their original objects. Native configuration is
validated separately. Changing parameters clears fitted state; inference-only
legacy forest archives cannot reconstruct training parameters for cloning.
"""
import functools
import inspect
import weakref
from types import GetSetDescriptorType

from ._array import Array
from ._buffer import _materialize, empty, as_f32_c
from ._arrays import _addr, _addr_ro
from . import _native_trees_bridge

# The compiled tree-ensemble owner: `score` predicts and scores in one native
# call (bindings/trees_public_score.mojo).
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()


_FOREST_ARRAYS = ("_offsets", "_colid", "_quesval", "_left_child", "_leaves")


class _ResidentForest:
    """Own a device snapshot independently of estimator lifetime or pickling."""
    def __init__(self, native, arrays, dimensions, mode, ordered=None):
        self.native = native
        self.arrays = arrays
        self.dimensions = dimensions
        self.mode = mode
        self.ordered = ordered
        params = list(dimensions)
        if ordered is not None:
            # 1 strict increasing-tree kernel, 0 the 32-grove graph; None
            # leaves the binary's compiled default.
            params.append(1 if ordered else 0)
        self.handle = native.forest_prepare_gpu(
            *(_addr_ro(a) for a in arrays), params)  # glue: one address per model array (five)
        self._finalizer = weakref.finalize(self, native.forest_release_gpu, self.handle)

    def matches(self, native, arrays, dimensions, mode, ordered=None):
        return (native is self.native and dimensions == self.dimensions and mode == self.mode
                and ordered == self.ordered
                and all(a is b for a, b in zip(arrays, self.arrays)))  # glue: compares the five model arrays


def forest_estimator(kind):
    """Register the explicit public constructor, including the mode wrapper."""
    def decorate(cls):
        original = cls.__init__
        signature = inspect.signature(original)
        parameters = list(signature.parameters.values())
        if "numeric_mode" not in signature.parameters:  # NumericModeMixin may have named it already
            parameters.append(inspect.Parameter("numeric_mode", inspect.Parameter.KEYWORD_ONLY,
                                                default=None))
        signature = signature.replace(parameters=parameters)

        # Import-time argument/layout metadata; binding a public call must not
        # execute inspect.Signature or inspect.Parameter Python methods.
        rows = tuple((p.name, int(p.kind), p.default is inspect.Parameter.empty,
                      None if p.default is inspect.Parameter.empty else p.default)
                     for p in parameters if p.name != "self")  # glue: fixed constructor signature
        if any(row[1] not in (0, 1, 3) for row in rows):  # glue: fixed constructor signature kinds
            raise TypeError("native forest constructors require named parameters")
        ancestry = type.__dict__["__mro__"].__get__(cls)
        namespaces = tuple(type.__dict__["__dict__"].__get__(base) for base in ancestry)  # glue: import-time native storage metadata
        descriptor = next(row["__dict__"] for row in namespaces if "__dict__" in row)  # glue: locate the native instance dictionary
        if type(cls) is not type or type(descriptor) is not GetSetDescriptorType:
            raise TypeError("native forest constructors require native instance storage")
        protected = ("__getattribute__", "__getattr__", "__setattr__", "__del__")
        hooks = tuple((name, next((row[name] for row in namespaces if name in row), None))  # glue: import-time inherited hook lookup
                      for name in protected)  # glue: fixed inherited hook identities
        if "trees_unavailable" not in _TREES_CONTEXT:
            _TREES_CONTEXT["trees_constructors"][cls] = (rows, descriptor.__get__, hooks)

        @functools.wraps(original)
        def initialize(self, *args, **kwargs):
            return _TREES(self, "trees_forest_construct", (cls, args, kwargs), _TREES_CONTEXT)

        initialize.__signature__ = signature
        cls.__init__ = initialize
        cls._forest_original_init = original
        cls._parameter_names = tuple(p.name for p in parameters if p.name != "self")  # glue: collects constructor parameter names
        cls._estimator_type = kind
        return cls
    return decorate


def held_resident_forest(model):
    """The prepared resident snapshot in `model`'s holder dict, or None.

    Reads the holder directly: the public forests' native attribute lookup
    refuses Python data descriptors such as the `_resident_forest` property,
    so `getattr(model, "_resident_forest", None)` raises TypeError there."""
    holder = model.__dict__.get("_resident_holder")  # glue: the fit's holder dict
    return holder.get("forest") if holder else None  # glue: one keyed read


def _frozen_copy(a):
    """A read-only copy of a flat int32 or float32 model array."""
    return Array.from_buffer(memoryview(a.tobytes()).cast("i" if a.dtype == "<i4" else "f"))


class ForestProtocol:
    def get_params(self, deep=True):
        return _TREES(self, "trees_forest_get_params", (deep,), _TREES_CONTEXT)

    def set_params(self, **params):
        return _TREES(self, "trees_public_forest_set_params", (params,), _TREES_CONTEXT)

    def _validate_params(self):
        """The constructor checks, run at fit (scikit-learn's convention):
        refuses a bad parameter with the message the constructor used to
        raise, and derives the native configuration (`_cfg`) from the
        stored parameters, which stay as given."""
        params = {name: getattr(self, name) for name in self._parameter_names}  # glue: reads constructor parameters by name
        mode = params.get("numeric_mode")
        if mode is not None and (not isinstance(mode, str) or
                mode.strip().lower() not in ("fast", "identical")):
            raise ValueError("numeric_mode must be fast, identical or None")
        self._validate_inference_engine(params.get("inference_engine", "sequential"))
        self._forest_validating = True
        try:
            type(self)._forest_original_init(self, **params)
        finally:
            self.__dict__.pop("_forest_validating", None)
        for name, value in params.items():  # glue: restores constructor parameters verbatim
            setattr(self, name, value)

    def _refresh_config(self):
        replacement = type(self)(**self.get_params())
        # Checks first: a refused parameter leaves this instance as it was.
        replacement._validate_params()
        # sklearn owns this transient context while Pipeline.fit is active,
        # and the fit tracker (`_mode._track_fit`) its in-fit count.
        keep = {k: self.__dict__[k] for k in ("_parent_callback_ctx", "_mojolearn_fitting")  # glue: two bookkeeping attribute names
                if k in self.__dict__}
        # Once a new fit begins, old nodes/classes must not survive a failure.
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        self.__dict__.update(keep)

    @staticmethod
    def _validated_mode(mode):
        if mode is None:
            return None
        if not isinstance(mode, str) or mode.strip().lower() not in (
                "fast", "identical"):
            raise ValueError("numeric_mode must be fast, identical or None")
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
        if not isinstance(engine, str) or engine not in ("auto", "sequential", "parallel_groves"):
            raise ValueError("inference_engine must be 'auto', 'sequential' or 'parallel_groves'")
        return engine

    def _prediction_engine(self):
        """The engine this model PREDICTS AS, which is also what its archive
        records: 'sequential' or 'parallel_groves'.

        IDENTICAL `auto` is 'sequential' even where a resident snapshot
        serves it (`_ordered_resident_auto`): that snapshot runs the strict
        increasing-tree kernel, the sequential route's arithmetic, so the
        model's bits and its archive are the sequential ones on every
        vendor and on the CPU. Until 2026-09-22 it answered
        'parallel_groves', which wrote a groves archive (every RF lane's
        saved-model hash moved) and, through the compiled default, moved an
        explicit IDENTICAL `parallel_groves` model off the 32-grove fold it
        was recorded with."""
        engine = self._validate_inference_engine(getattr(self, "inference_engine", "sequential"))
        if engine == "auto":
            # FAST permits the grove-parallel reduction and benefits from a
            # persistent device snapshot.
            if self._effective_mode() == "fast":
                return "parallel_groves"
            return "sequential"
        return engine

    def _ordered_resident_auto(self):
        """The `sequential` engine on a GPU binding (one with the
        `forest_ordered_resident` export): predict through a device-resident
        snapshot prepared with the strict increasing-tree kernel, the
        sequential route's arithmetic (cpu-gpu-cleanup t-forest: the GPU
        install no longer walks the trees on host threads). A host binding
        (CPU-only install) has no such route; its proxy raises ImportError
        by name for any function it lacks, so a missing export is an absence
        here and the host walk runs."""
        if self._prediction_engine() != "sequential":
            return False
        return self._ordered_resident_export() is not None

    def _ordered_resident_export(self):
        """The binding's `forest_ordered_resident`, or None where it has none:
        an older binary, or a host binding, whose proxy raises ImportError
        by name (not AttributeError) for an export it lacks."""
        try:
            selected = getattr(self._bind(), "forest_ordered_resident", None)
        except (AttributeError, ImportError):
            return None
        return selected if callable(selected) else None

    def _resident_ordered_flag(self):
        """The aggregation a resident snapshot is prepared with: True strict
        increasing-tree, False the 32-grove graph, None the compiled
        default. The `sequential` engine (served resident by
        `_ordered_resident_auto` on a GPU binding) is always strict. IDENTICAL
        names the groves explicitly so the recorded bits do not depend on a
        vendor build default: `parallel_groves` is the grove
        fold that the CPU host groves engine and the recorded GPU columns
        compute. FAST keeps the binary's default, and so
        does a binary older than the per-snapshot choice (no
        `forest_ordered_resident` export), whose resident route is groves."""
        if self._ordered_resident_export() is None:
            return None
        if self._prediction_engine() == "sequential":
            return True
        if self._effective_mode() != "identical":
            return None
        return False

    def _forest_auxiliary(self, X, *, leaves=False, prefixes=False, prediction=False):
        """T39 buffer/argument plumbing; all walks and reductions are Mojo."""
        return _TREES(self, "trees_public_forest_auxiliary", (X, leaves, prefixes, prediction), _TREES_CONTEXT)

    def apply(self, X):
        """Return tree-relative leaf IDs, shape (rows, trees), from the native walk."""
        return _TREES(self, "trees_public_forest_apply", (X,), _TREES_CONTEXT)

    def predict_with_leaves(self, X):
        """Return ordered raw mean scores (rows, outputs) and leaf IDs (rows, trees).

        Classifier scores are undecoded class votes. This auxiliary ordered
        contract is independent of an explicitly selected groves engine.
        """
        return _TREES(self, "trees_public_forest_predict_with_leaves", (X,), _TREES_CONTEXT)

    def staged_predict(self, X):
        """Return eager mean prefixes, shape (rows, trees, outputs), in tree order.

        Each prefix uses the fixed increasing-tree fold through that tree.
        Only this requested API allocates the complete prefix buffer.
        """
        return _TREES(self, "trees_public_forest_staged_predict", (X,), _TREES_CONTEXT)

    def _prepare_resident_forest(self, native=None):
        """Prepare the existing immutable parallel-groves snapshot without a query."""
        if self._prediction_engine() != "parallel_groves" and not self._ordered_resident_auto():
            raise ValueError("resident forest preparation requires parallel_groves")
        if not hasattr(self, "_offsets"):
            raise RuntimeError("this estimator is not fitted yet")
        if native is None:
            native = self._bind()
        dimensions = (int(self.n_features_in_), int(self._n_trees), int(self._num_outputs))
        arrays = tuple(getattr(self, name) for name in _FOREST_ARRAYS)  # glue: the five model arrays
        required = ("forest_prepare_gpu", "forest_predict_resident_reuse_gpu", "forest_release_gpu")
        if any(not callable(getattr(native, name, None)) for name in required):  # glue: checks three binding entry points
            raise RuntimeError("rebuild the forest binding for resident parallel_groves inference")
        mode = self._effective_mode()
        ordered = self._resident_ordered_flag()
        resident = held_resident_forest(self)
        if resident is None or not resident.matches(native, arrays, dimensions, mode, ordered):
            # A bytes owner cannot be made writable again via setflags(). A
            # caller retaining an old mutable private-array alias cannot alter
            # the device snapshot or the host model used by save/sequential.
            dtypes = ("<i4", "<i4", "<f4", "<i4", "<f4")
            arrays = tuple(_materialize(a, "forest model")[0] for a in arrays)  # glue: materializes the five model arrays
            for a, dtype in zip(arrays, dtypes):  # glue: checks the five model arrays
                if not isinstance(a, Array) or a.dtype != dtype or a.ndim != 1:
                    raise ValueError("forest model arrays must have their original flat dtypes")
            nodes = arrays[1].size
            if (arrays[0].size != dimensions[1] + 1 or nodes < 1
                    or arrays[2].size != nodes or arrays[3].size != nodes
                    or arrays[4].size != nodes * dimensions[2]
                    or int(arrays[0][-1]) != nodes):
                raise ValueError("forest model array shapes do not match metadata")
            # A fit leaves its arrays frozen already (`_mojolearn_after_fit`),
            # so a predict after a fit stores nothing new on the instance.
            frozen = tuple(a if not a.flags["WRITEABLE"] else _frozen_copy(a) for a in arrays)  # glue: freezes the five model arrays
            resident = _ResidentForest(native, frozen, dimensions, mode, ordered)
            for name, a in zip(_FOREST_ARRAYS, frozen):  # glue: stores the five model arrays
                if getattr(self, name) is not a:
                    setattr(self, name, a)
            self._resident_forest = resident
        return resident

    # The device snapshot lives in a holder dict the fit creates, not in an
    # attribute of its own: a predict that prepares the snapshot changes
    # the holder's contents, never `__dict__` (scikit-learn's
    # check_dict_unchanged compares `__dict__` around predict).
    @property
    def _resident_forest(self):
        holder = self.__dict__.get("_resident_holder")
        if not holder or "forest" not in holder:
            raise AttributeError("_resident_forest")
        return holder["forest"]

    @_resident_forest.setter
    def _resident_forest(self, value):
        holder = self.__dict__.get("_resident_holder")
        if holder is None:
            holder = self.__dict__["_resident_holder"] = {}
        holder["forest"] = value

    def __getstate__(self):
        return _TREES(self, "trees_forest_getstate", (), _TREES_CONTEXT)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__["_resident_holder"] = {}

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_offsets")

    def __sklearn_tags__(self):
        """scikit-learn's tag protocol, read only by scikit-learn: the shared
        `_mode.ParamsMixin` answer (estimator type, required target, the
        classifier or regressor tags), which is what this protocol returned."""
        from ._mode import ParamsMixin
        return ParamsMixin.__sklearn_tags__(self)

    def score(self, X, y, sample_weight=None):
        """Mode-aware GPU accuracy or Float32 R², weighted when `sample_weight`
        is given (scikit-learn's weighted accuracy and R² on the pinned-sum
        path of `metrics.accuracy_score` and `metrics.r2_score`).

        Class labels are compared without narrowing their values by the
        x_metrics binding's grouped match count (`accuracy_fraction`).
        Regression scoring uses the same Float32 target domain as tree fitting.
        """
        return _TREES(self, "trees_public_forest_score", (X, y, sample_weight), _TREES_CONTEXT)


