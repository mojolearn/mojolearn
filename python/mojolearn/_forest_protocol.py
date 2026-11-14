# SPDX-License-Identifier: Apache-2.0
"""Bounded sklearn protocol for RF/ET; sklearn remains an optional dependency.

Constructor parameters retain their original objects. Native configuration is
validated separately. Changing parameters clears fitted state; inference-only
legacy forest archives cannot reconstruct training parameters for cloning.
"""
import functools
import inspect
from types import GetSetDescriptorType

from . import _native_trees_bridge

# The compiled tree-ensemble owner: `score` predicts and scores in one native
# call (bindings/trees_public_score_door.mojo).
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()


# `_ResidentForest` (a Python-owned device snapshot) deleted by lane py-trees
# (2026-11-14): no product path built one; the forest door keeps snapshots in
# its native resident table (bindings/py_trees_resident.mojo).


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


class ForestProtocol:
    def get_params(self, deep=True):
        return _TREES(self, "trees_forest_get_params", (deep,), _TREES_CONTEXT)

    def set_params(self, **params):
        return _TREES(self, "trees_public_forest_set_params", (params,), _TREES_CONTEXT)

    def _validate_params(self):
        """The constructor checks, run at fit (scikit-learn's convention):
        refuses a bad parameter with the constructor's message and stores the
        derived native configuration (`_cfg`); the parameters stay as given."""
        return _TREES(self, "trees_public_forest_validate_params", (), _TREES_CONTEXT)

    def _refresh_config(self):
        """The constructor checks on the stored parameters (a refused one
        leaves the instance as it was), then the state a new constructor plus
        `_validate_params` would hold; fitted state is dropped."""
        return _TREES(self, "trees_public_forest_refresh", (), _TREES_CONTEXT)

    def _capture_fit_mode(self):
        return _TREES(self, "trees_public_forest_capture_mode", (), _TREES_CONTEXT)

    def _effective_mode(self):
        return _TREES(self, "trees_public_forest_effective_mode", (), _TREES_CONTEXT)

    def _bind(self, name=None):
        from . import _backend
        return _backend.binding(name or self._BINDING, self._effective_mode())

    def _prediction_engine(self):
        """The engine this model PREDICTS AS, which is also what its archive
        records: 'sequential' or 'parallel_groves'. IDENTICAL `auto` is
        'sequential'; FAST `auto` is 'parallel_groves'."""
        return _TREES(self, "trees_public_forest_prediction_engine", (), _TREES_CONTEXT)

    def _ordered_resident_auto(self):
        """True when the sequential engine predicts through a device-resident
        snapshot prepared with the strict increasing-tree kernel (an image
        whose snapshot takes the ordered word); False on a host binding."""
        return _TREES(self, "trees_public_forest_ordered_resident_auto", (), _TREES_CONTEXT)

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
        return _TREES(self, "trees_public_forest_setstate", (state,), _TREES_CONTEXT)

    def __sklearn_is_fitted__(self):
        return _TREES(self, "trees_public_forest_is_fitted", (), _TREES_CONTEXT)

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


