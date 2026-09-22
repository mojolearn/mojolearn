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
    a default you can set in code  mojolearn.set_numeric_mode("deterministic")
    a per-estimator parameter      RandomForestClassifier(numeric_mode=...)

The environment variable still works and still sets the STARTING default, so
nothing written against the old spelling breaks.

WHY THIS IS AN ATTRIBUTE LOOKUP AND NOT A REBIND. **The three tiers are one
source under one flag, not three implementations.** `GLOBAL_NUMERIC_MODE` in
`checks/numerics.mojo` is that flag; `PIN_DETERMINISM` and
`PIN_CROSS_VENDOR` derive from it and every kernel reads those. But the flag
is COMPTIME, which is what lets the fast build carry none of the pinning code
at all rather than branching past it at run time, so the three settings are
compiled and shipped side by side and a Python parameter selects among them
rather than flipping anything inside one of them. That is
`_backend.load_set`, and every call site therefore has to ask for its binding
at CALL TIME rather than binding a module-level name at import.

THAT THE TIERS COEXIST IS MEASURED (2026-08-29). Each `.so` carries its own
Mojo runtime and opens its own device context, so "two of them in one process
will fight" was the real risk. All three were loaded together and then called
INTERLEAVED -- fast, deterministic, identical, fast -- twice, on one
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


class ParamsMixin:
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
        names = [p.name for p in params if p.name != "self"
                 and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if issubclass(cls, NumericModeMixin) and "numeric_mode" not in names:
            names.append("numeric_mode")
        return sorted(names)

    def get_params(self, deep=True):
        out = {}
        for name in self._get_param_names():
            try:
                value = getattr(self, name)
            except AttributeError:
                raise TypeError(
                    f"mojolearn {type(self).__name__}: constructor parameter "
                    f"{name!r} is not stored under its own name, so this "
                    "estimator cannot report or clone its parameters"
                ) from None
            if deep and not isinstance(value, type) and callable(getattr(value, "get_params", None)):
                out.update((f"{name}__{k}", v) for k, v in value.get_params().items())
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
        for key, value in params.items():
            name, sep, sub = key.partition("__")
            if name not in values:
                raise ValueError(
                    f"Invalid parameter {name!r} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(values)!r}."
                )
            if sep:
                nested.setdefault(name, {})[sub] = value
            else:
                values[name] = value
        for name, sub in nested.items():
            values[name].set_params(**sub)
        replacement = type(self)(**values)
        self.__dict__.clear()
        self.__dict__.update(replacement.__dict__)
        return self

    def __sklearn_tags__(self):
        """scikit-learn's tag protocol (1.6+), read only by scikit-learn,
        so importing it here needs no dependency: `_estimator_type` says
        classifier or regressor, which picks stratified folds and the
        default scorer in scikit-learn's own `cross_val_score`."""
        from sklearn.utils import ClassifierTags, RegressorTags, Tags, TargetTags
        kind = getattr(self, "_estimator_type", None)
        return Tags(estimator_type=kind,
                    target_tags=TargetTags(required=kind in ("classifier", "regressor")),
                    classifier_tags=ClassifierTags() if kind == "classifier" else None,
                    regressor_tags=RegressorTags() if kind == "regressor" else None)


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
        for method_name in ("fit", "partial_fit", "fit_predict", "fit_transform", "_fit_with_tree_start"):
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

        cls.__init__ = __init__

    #: The binding this estimator family talks to. Subclasses that use more
    #: than one pass the name explicitly.
    _BINDING = "_mojolearn"

    def _bind(self, name=None):
        return _backend.binding(
            name or self._BINDING, getattr(self, "numeric_mode", None)
        )

    def numeric_mode_used(self):
        """The tier THIS estimator will run on, resolved and read back from
        the binary it actually holds -- not the string that was passed in."""
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

    def vendor_used(self):
        """'metal', 'cuda' or 'hip': the accelerator API of the binary THIS
        estimator will call, read back from that binary's own compile-time
        constant (`<prefix>_vendor()`, `checks/vendor.mojo`), not from
        the directory it was loaded from and not from the platform. None
        for a binary built before the read-back existed."""
        return _backend.read_vendor(self._bind())
