# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`numeric_mode=` as a PARAMETER, on the estimator, in your code.

WHAT THIS REPLACES. The mode used to be reachable only as
`MOJOLEARN_NUMERIC_MODE`, an environment variable read ONCE by
`_backend.select()` before the first estimator was imported. One install, but
a global set outside the program: you could not change it after import, and
you could not give two estimators in one script two different modes. The
library was shipping a choice you had to make from the shell.

It is now three things, and only the first is unchanged:

    ONE install                    the wheel carries every tier's binaries
    a default you can set in code  mojolearn.set_numeric_mode("identical")
    a per-estimator parameter      RandomForestClassifier(numeric_mode=...)

The environment variable still works and still sets the STARTING default, so
nothing written against the old spelling breaks.

WHY THIS IS AN ATTRIBUTE LOOKUP AND NOT A REBIND. **The tiers are one
source under one flag, not separate implementations.** `GLOBAL_NUMERIC_MODE` in
`checks/numerics.mojo` is that flag; `PIN_DETERMINISM` and
`PIN_CROSS_VENDOR` derive from it and every kernel reads those. But the flag
is COMPTIME, which is what lets the fast build carry none of the pinning code
at all rather than branching past it at run time, so the settings are
compiled and shipped side by side and a Python parameter selects among them
rather than flipping anything inside one of them. That is
`_backend.load_set`, and every call site therefore has to ask for its binding
at CALL TIME rather than binding a module-level name at import.

THAT THE TIERS COEXIST IS MEASURED (2026-08-29). Each `.so` carries its own
Mojo runtime and opens its own device context, so "two of them in one process
will fight" was the real risk. They were loaded together and then called
INTERLEAVED -- fast, identical, fast -- twice, on one
256x4096 @ 4096x128 product on an Apple M4. Each returned its own arithmetic
every time, and a call made after the identical set did not inherit its
answer. See `_backend.load_set`'s header.
"""

import functools
import inspect

from . import _backend


def _guard_cpu_training(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        from ._cpu_reference import require_training
        require_training(self)
        return method(self, *args, **kwargs)
    return guarded


def sklearn_tags(est):
    """scikit-learn's tag protocol (1.6+) for a mojolearn estimator, read only
    by scikit-learn, so importing it here needs no dependency.

    One answer for every public estimator, built from class attributes:

    - `_estimator_type`: 'classifier', 'regressor', 'clusterer',
      'outlier_detector', 'density_estimator' or None, as scikit-learn's
      mixins set it. It picks stratified folds and the default scorer in
      scikit-learn's `cross_val_score` and answers `is_classifier`.
    - a `transform` method gives transformer tags. `_sklearn_preserves_dtype`
      names the dtypes `transform` keeps (default: none claimed).
    - `_sklearn_input_tags` / `_sklearn_target_tags`: overrides of the
      matching scikit-learn tag fields (allow_nan, positive_only,
      one_d_array, ...), set where the estimator's contract differs from
      the defaults.
    - `_sklearn_requires_fit = False` for a stateless transformer.
    """
    from sklearn.utils import (ClassifierTags, InputTags, RegressorTags,  # glue: sklearn tag protocol, imported only when sklearn asks
                               Tags, TargetTags, TransformerTags)
    cls = type(est)
    kind = getattr(est, "_estimator_type", None)
    tags = Tags(estimator_type=kind,
                target_tags=TargetTags(required=kind in ("classifier", "regressor")),
                input_tags=InputTags(),
                classifier_tags=ClassifierTags() if kind == "classifier" else None,
                regressor_tags=RegressorTags() if kind == "regressor" else None,
                requires_fit=getattr(cls, "_sklearn_requires_fit", True))
    if callable(getattr(est, "transform", None)):
        tags.transformer_tags = TransformerTags(
            preserves_dtype=list(getattr(cls, "_sklearn_preserves_dtype", ())))
    for name, value in getattr(cls, "_sklearn_input_tags", {}).items():  # glue: sklearn tag field overrides by name
        setattr(tags.input_tags, name, value)
    for name, value in getattr(cls, "_sklearn_target_tags", {}).items():  # glue: sklearn tag field overrides by name
        setattr(tags.target_tags, name, value)
    return tags


#: The methods that need a fitted estimator. Called before `fit` they raise
#: `mojolearn.exceptions.NotFittedError`, as scikit-learn's estimators do.
_NEEDS_FIT = ("predict", "predict_proba", "predict_log_proba", "decision_function", "transform",
              "inverse_transform", "score", "score_samples")

#: The methods that fit. A completed call marks the instance fitted.
_FITS = ("fit", "partial_fit", "fit_predict", "fit_transform", "_fit_with_tree_start")

#: The constructors of a fitted instance from a saved model.
_LOADS = ("load", "from_checkpoint")


def is_fitted(est):
    """True once `fit` (or a loader) has completed on `est`.

    The flag is set by the `fit` wrapper below. An instance without it (one
    pickled before the flag existed, or assembled from parts) counts as
    fitted when it carries a public attribute ending in an underscore that
    is not None (scikit-learn's `check_is_fitted` rule; a constructor that
    sets `model_ = None` placeholders does not count). While a `fit` is running on the
    instance, its own calls to `predict` or `transform` are let through.
    """
    state = vars(est)
    if state.get("_mojolearn_fitted") or state.get("_mojolearn_fitting"):
        return True
    return any(k.endswith("_") and not k.startswith("_") and v is not None for k, v in state.items())  # glue: scans fitted attribute names


def _track_fit(method):
    if getattr(method, "_mojolearn_tracks_fit", False):
        return method

    @functools.wraps(method)
    def tracked(self, *args, **kwargs):
        state = vars(self)
        state["_mojolearn_fitting"] = state.get("_mojolearn_fitting", 0) + 1
        try:
            out = method(self, *args, **kwargs)
        finally:
            # `vars(self)` again: `set_params` inside a fit replaces the dict's contents
            state = vars(self)
            depth = state.get("_mojolearn_fitting", 1) - 1
            if depth > 0:
                state["_mojolearn_fitting"] = depth
            else:
                state.pop("_mojolearn_fitting", None)
        state["_mojolearn_fitted"] = True
        return out

    tracked._mojolearn_tracks_fit = True
    return tracked


def _needs_fit(method, name):
    if getattr(method, "_mojolearn_needs_fit", False):
        return method

    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if not is_fitted(self):
            from .exceptions import not_fitted_error
            raise not_fitted_error(
                f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with "
                f"appropriate arguments before using {name}.")
        return method(self, *args, **kwargs)

    guarded._mojolearn_needs_fit = True
    return guarded


def _loader(method):
    @functools.wraps(method)
    def load(cls, *args, **kwargs):
        out = method(cls, *args, **kwargs)
        if isinstance(out, cls):
            vars(out)["_mojolearn_fitted"] = True
        return out

    return classmethod(load)


def install_fit_tracking(cls):
    """Mark `cls` fitted after its fits and loaders; refuse its predict,
    transform and score methods before that (see `is_fitted`). A class with
    `_sklearn_requires_fit = False` (a stateless transformer) is not guarded.
    """
    for name in _FITS:  # glue: wraps the fit method names
        method = cls.__dict__.get(name)
        if inspect.isfunction(method):
            setattr(cls, name, _track_fit(method))
    for name in _LOADS:  # glue: wraps the loader method names
        method = cls.__dict__.get(name)
        if isinstance(method, classmethod):
            setattr(cls, name, _loader(method.__func__))
    if not getattr(cls, "_sklearn_requires_fit", True):
        for name in _NEEDS_FIT:  # glue: unwraps the method names that need a fit
            method = getattr(cls, name, None)
            if getattr(method, "_mojolearn_needs_fit", False):  # guarded on a stateful base
                setattr(cls, name, method.__wrapped__)
        return
    for name in _NEEDS_FIT:  # glue: wraps the method names that need a fit
        method = getattr(cls, name, None)
        if inspect.isfunction(method) and not getattr(method, "_mojolearn_needs_fit", False):
            setattr(cls, name, _needs_fit(method, name))


class SklearnTagsMixin:
    """`__sklearn_tags__` for an estimator: see `sklearn_tags`. Subclasses
    get fit tracking and `NotFittedError` (see `install_fit_tracking`)."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        install_fit_tracking(cls)

    def __sklearn_tags__(self):
        return sklearn_tags(self)

    def __sklearn_is_fitted__(self):
        return is_fitted(self)


class ClassifierScoreMixin:
    """scikit-learn's `ClassifierMixin.score` for a classifier without its own:
    the default scorer of `GridSearchCV` and `cross_val_score`."""

    def score(self, X, y, sample_weight=None):
        """Mean accuracy of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            Fraction of samples predicted correctly (weighted when
            ``sample_weight`` is given).
        """
        # The match count and weight total are the x_metrics binding's sums.
        from ._expansion_metrics import accuracy_fraction
        return accuracy_fraction(y, self.predict(X), sample_weight,
                                 getattr(self, "numeric_mode", None))


class RegressorScoreMixin:
    """scikit-learn's `RegressorMixin.score` for a regressor without its own:
    the default scorer of `GridSearchCV` and `cross_val_score`."""

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        # R^2 is the metrics binding's pinned-sum r2_score, in float32.
        from ._buffer import as_f32_c
        from ._metrics_impl import r2_score
        truth, _ = as_f32_c(y, ndim=None, name="y")
        pred, _ = as_f32_c(self.predict(X), ndim=None, name="prediction")
        return r2_score(truth, pred, sample_weight=sample_weight,
                        numeric_mode=getattr(self, "numeric_mode", None))


class ParamsMixin(SklearnTagsMixin):
    """`get_params` / `set_params` read off the constructor's signature.

    scikit-learn's convention (base.py `BaseEstimator.get_params`): every
    constructor parameter is stored on the instance under its own name, so
    the parameters ARE the attributes the signature names. Without these,
    `mojolearn.cross_val_score(mojolearn.Ridge(), X, y)` refused its own
    library's estimator ("cross_val_score estimator must implement
    get_params"), and so did scikit-learn's `clone` (Sep 22 pip smoke).
    A class that defines its own `get_params` (the forests, the scalers)
    keeps it: those come earlier in the MRO.
    """

    @classmethod
    def _get_param_names(cls):
        init = cls.__init__
        if init is object.__init__:
            return []
        # `inspect.signature` follows `__wrapped__` (NumericModeMixin's
        # wrapper), so this is the class's own constructor signature.
        params = inspect.signature(init).parameters.values()
        names = [p.name for p in params if p.name != "self"  # glue: sklearn get_params parameter names
                 and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if issubclass(cls, NumericModeMixin) and "numeric_mode" not in names:
            names.append("numeric_mode")
        return sorted(names)  # glue: sklearn get_params parameter names

    def get_params(self, deep=True):
        out = {}
        for name in self._get_param_names():  # glue: sklearn get_params parameter names
            try:
                value = getattr(self, name)
            except AttributeError:
                raise TypeError(
                    f"mojolearn {type(self).__name__}: constructor parameter "
                    f"{name!r} is not stored under its own name, so this "
                    "estimator cannot report or clone its parameters"
                ) from None
            if deep and not isinstance(value, type) and callable(getattr(value, "get_params", None)):
                out.update((f"{name}__{k}", v) for k, v in value.get_params().items())  # glue: sklearn nested get_params plumbing
            out[name] = value
        return out

    def set_params(self, **params):
        """Set constructor parameters. The estimator is rebuilt through its
        constructor, so every constructor check runs again and any fitted
        state is dropped, as a new setting invalidates it."""
        if not params:
            return self
        values = self.get_params(deep=False)
        nested = {}
        for key, value in params.items():  # glue: sklearn set_params keyword arguments
            name, sep, sub = key.partition("__")  # glue: splits a parameter name string
            if name not in values:
                raise ValueError(
                    f"Invalid parameter {name!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(values)!r}."
                )
            if sep:
                nested.setdefault(name, {})[sub] = value
            else:
                values[name] = value
        for name, sub in nested.items():  # glue: sklearn nested set_params plumbing
            values[name].set_params(**sub)
        replacement = type(self)(**values)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self


class NumericModeMixin(ParamsMixin):
    """Gives an estimator `self._bind(name)`.

    `numeric_mode` is read off the instance at every call rather than
    resolved in `__init__`, so an estimator that is unpickled from an older
    version (no such attribute) falls back to the process default instead of
    raising -- and so that setting the attribute after construction works the
    way a caller would expect it to.
    """

    def __init_subclass__(cls, **kw):
        """Give every estimator a `numeric_mode=` keyword without editing
        eleven constructor signatures.

        WHY A WRAPPER AND NOT ELEVEN EDITS. The parameter is identical in
        every class and is not part of any estimator's own contract -- it
        selects which BINARY answers, which is a property of the library, not
        of k-means. Eleven hand-written copies is eleven chances for one of
        them to drift, and the drift would be silent: an estimator that
        quietly ignored the keyword would run on the process default and
        report a tier it was not using.

        The assignment happens AFTER the wrapped `__init__`, because these
        classes inherit (`ExtraTreesClassifier` -> `_ExtraTreesBase`,
        `KNeighborsClassifier` -> `NearestNeighbors`) and BOTH ends get
        wrapped. The base runs first and would otherwise write its own
        `None` over the subclass's real answer.
        """
        super().__init_subclass__(**kw)
        # Guard even passive fits (e.g. k-NN/KDE storing samples). Inherited
        # fit methods retain their guard, including explicit host subclasses
        # used on a machine that also has a GPU. `_fit_with_tree_start` is
        # the forests' shard fit, which `parallel_ensemble.fit_forest`'s
        # worker calls without `fit`; since the rf and trees host bindings
        # serve it (lane/cpu-training-par-wave2, 2026-09-15) it is guarded
        # like `fit`, so a CPU-only install trains a shard only inside
        # `reference_training()`.
        for method_name in ("fit", "partial_fit", "fit_predict", "fit_transform", "_fit_with_tree_start"):  # glue: wraps five estimator method names
            method = cls.__dict__.get(method_name)
            if method is not None:
                setattr(cls, method_name, _guard_cpu_training(method))
        orig = cls.__dict__.get("__init__")
        if orig is None:
            return

        @functools.wraps(orig)
        def __init__(self, *args, numeric_mode=None, **kwargs):
            orig(self, *args, **kwargs)
            if numeric_mode is not None or not hasattr(self, "numeric_mode"):
                self.numeric_mode = numeric_mode

        # The signature scikit-learn reads (`_get_param_names`, its
        # `check_no_attributes_set_in_init`) names `numeric_mode` too.
        sig = inspect.signature(orig)
        if "numeric_mode" not in sig.parameters:
            params = list(sig.parameters.values())
            at = next((i for i, p in enumerate(params) if p.kind is p.VAR_KEYWORD), len(params))  # glue: signature parameter position
            params.insert(at, inspect.Parameter("numeric_mode", inspect.Parameter.KEYWORD_ONLY, default=None))
            sig = sig.replace(parameters=params)
        __init__.__signature__ = sig
        cls.__init__ = __init__

    #: The binding this estimator family talks to. Subclasses that use more
    #: than one pass the name explicitly.
    _BINDING = "_mojolearn"

    def _bind(self, name=None):
        return _backend.binding(
            name or self._BINDING, getattr(self, "numeric_mode", None)
        )

    def numeric_mode_used(self):
        """The numeric mode this estimator runs in.

        Returns
        -------
        str
            ``'identical'`` or ``'fast'``, read from the compiled
            binary the estimator uses, not from the argument it was given.
        """
        module = self._bind()
        name = module.__name__.rsplit(".", 1)[-1]
        getter_name = _backend._vendor_fn(name).removesuffix("_vendor") + "_numeric_mode"
        getter = getattr(module, getter_name, None)
        # Older metrics modules expose the same compile-time constant under
        # UMAP's original name. A DSO's __name__ need not retain its tier path.
        if getter is None and name == "_mojolearn_metrics":
            getter = getattr(module, "umap_numeric_mode", None)
        if getter is not None:
            code = getter()
            if code not in _backend._CODE_MODE:
                raise RuntimeError(f"mojolearn: {name} returned unknown numeric mode {code!r}")
            return _backend._CODE_MODE[code]
        # Legacy artifacts without readback retain the historical path hint.
        return module.__name__.split(".")[-2]

    # Read back from the binary's own compile-time constant (`<prefix>_vendor()`, checks/vendor.mojo), not
    # from the directory it was loaded from or the platform; None for a binary built before the read-back.
    def vendor_used(self):
        """The GPU backend this estimator runs on.

        Returns
        -------
        str
            ``'metal'``, ``'cuda'`` or ``'hip'``, read from the compiled binary the
            estimator uses.
        """
        return _backend.read_vendor(self._bind())
