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
import sys

from . import _backend
from ._array import Array as _Array
from ._device_table import CPU_CLASSES as _DEVICE_CPU_CLASSES, record as _device_record
from . import _sk_validate
from ._digest_fields import CONNECTION as _DIGEST_CONNECTION, declare as _digest_declare, read as _digest_read
from ._digest_fields import TRAINING_ARGUMENTS as _DIGEST_TRAINING_ARGUMENTS


def _guard_cpu_training(method):
    # CPU training is public: nothing refuses it, so the former per-fit wrapper is gone and nothing calls this
    # any more (its class-definition loop in `_install_numeric_mode` set every
    # fit method to itself; deleted by lane py-mode-wrappers, 2026-11-15).
    # Kept only as the fixed warning location: `_native_classical_warnings`
    # reads this function's final line as the `mojolearn/_mode.py:<line>` the
    # native warnings report.
    return method


def sklearn_tags(est, preserves_dtype=None):
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
    - `_sklearn_instance_target_tags()`: target tag overrides that depend
      on the instance's parameters.
    - `_sklearn_requires_fit = False` for a stateless transformer.

    `preserves_dtype` fixes the transformer tags' dtype names for an
    estimator whose `transform` always keeps them (the float32 scalers).
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
        if preserves_dtype is not None:
            tags.transformer_tags.preserves_dtype = preserves_dtype
    from ._sk_validate import input_tag_overrides
    for name, value in input_tag_overrides(cls).items():  # glue: sklearn tag field overrides by name
        setattr(tags.input_tags, name, value)
    for name, value in getattr(cls, "_sklearn_input_tags", {}).items():  # glue: sklearn tag field overrides by name
        setattr(tags.input_tags, name, value)
    for name, value in getattr(cls, "_sklearn_target_tags", {}).items():  # glue: sklearn tag field overrides by name
        setattr(tags.target_tags, name, value)
    # target tags that depend on a parameter (TweedieRegressor's power)
    own = getattr(est, "_sklearn_instance_target_tags", None)
    if callable(own):
        for name, value in own().items():  # glue: sklearn tag field overrides by name
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
    return _MODE_METADATA(est, "is_fitted", _MODE_METADATA_CONTEXT)


def _track_fit(method):
    if getattr(method, "_mojolearn_tracks_fit", False):
        return method
    # scikit-learn's input and target validation (`_sk_validate`) runs on the
    # outermost fit only: a fit's own nested fits see validated input.
    validates = method.__name__ in _sk_validate.FIT_METHODS
    slots = _sk_validate.arg_slots(method) if validates else (None, None, None)

    @functools.wraps(method)
    def tracked(self, *args, **kwargs):
        state = vars(self)
        check = None
        if validates and not state.get("_mojolearn_fitting"):
            args, kwargs, check = _sk_validate.before_fit(self, method.__name__, slots, args, kwargs)
        state["_mojolearn_fitting"] = state.get("_mojolearn_fitting", 0) + 1
        try:
            out = method(self, *args, **kwargs)
        except Exception as err:
            if check is not None:
                _sk_validate.reraise(self, check, err)
            raise
        finally:
            # `vars(self)` again: `set_params` inside a fit replaces the dict's contents
            state = vars(self)
            depth = state.get("_mojolearn_fitting", 1) - 1
            if depth > 0:
                state["_mojolearn_fitting"] = depth
            else:
                state.pop("_mojolearn_fitting", None)
        state["_mojolearn_fitted"] = True
        # A fit leaves every per-instance cache slot its predict path uses
        # in place, so predict and transform never add to `__dict__`
        # (scikit-learn's check_dict_unchanged): see `_mojolearn_after_fit`.
        after = getattr(self, "_mojolearn_after_fit", None)
        if after is not None:
            after()
        if check is not None:
            _sk_validate.after_fit(self, check)
        return out

    tracked._mojolearn_tracks_fit = True
    return tracked


def _needs_fit(method, name, require_fit=True):
    """`method` refusing an unfitted instance (unless `require_fit` is False,
    a stateless transformer), then validating its X (`_sk_validate`)."""
    if getattr(method, "_mojolearn_needs_fit", False):
        return method
    slots = _sk_validate.arg_slots(method)

    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        if require_fit and not is_fitted(self):
            from .exceptions import _not_fitted_error
            raise _not_fitted_error(
                f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with "
                f"appropriate arguments before using {name}.")
        if vars(self).get("_mojolearn_fitting"):
            return method(self, *args, **kwargs)
        args, kwargs, check = _sk_validate.before_call(self, name, slots, args, kwargs)
        if check is None:
            return method(self, *args, **kwargs)
        try:
            out = method(self, *args, **kwargs)
        except Exception as err:
            _sk_validate.reraise(self, check, err)
            raise
        if not _caller_outside(sys._getframe(1)):
            # mojolearn's own chains (OneClassSVM.predict -> decision_function
            # -> score_samples) pass these Arrays straight to float32 kernels:
            # a float64 widening here was read back as float32 words, so each
            # label came from the bytes of another row's score.
            return out
        return _sk_validate.after_call(self, name, out)

    guarded._mojolearn_needs_fit = require_fit
    guarded._mojolearn_validates = True
    return guarded


def _loader(method):
    # Only an application subclass's loaders reach this (`install_fit_tracking`
    # runs for application subclasses only). Every package loader is the one
    # archive binding call, and the native loader publishes `_mojolearn_fitted`
    # itself (bindings/host_archive_native_door.mojo `host_archive_load` for the
    # registered classes, `host_archive_public_forest` for RF/ET/DT; the GBDT
    # adapters' load is a native refusal), so no package `load` runs this
    # Python frame (lane py-mode-wrappers, 2026-11-15).
    if getattr(method, "_mojolearn_tracks_fit", False):
        # The native archive owner publishes fitted state and admits any
        # after-fit hook before returning; no Python post-load callback.
        return classmethod(method)
    @functools.wraps(method)
    def load(cls, *args, **kwargs):
        out = method(cls, *args, **kwargs)
        if isinstance(out, cls):
            vars(out)["_mojolearn_fitted"] = True
            after = getattr(out, "_mojolearn_after_fit", None)
            if after is not None:
                after()
        return out

    load._mojolearn_loader = True
    return classmethod(load)


def _application_subclass(cls):
    """A class written outside the package (or in its tests): its own
    overrides are ordinary application code and keep fit tracking, fitted
    checks and NumPy outputs (`install_fit_tracking`). A package class is
    never wrapped: each of its public methods is the native owner's one
    binding call, which owns validation, fit tracking, the fitted check and
    the public result itself."""
    module = getattr(cls, "__module__", "") or ""
    return ("_mojolearn_native_public" not in cls.__dict__
            and (not module.startswith("mojolearn.") or module.startswith("mojolearn.tests")))


def install_fit_tracking(cls, own_only=False):
    """Mark `cls` fitted after its fits and loaders; refuse its predict,
    transform and score methods before that (see `is_fitted`). A class with
    `_sklearn_requires_fit = False` (a stateless transformer) is not guarded.
    Also installs the NumPy outputs (`install_public_outputs`).

    Called for application subclasses only (`SklearnTagsMixin`); a package
    class keeps its one-call native methods unwrapped.
    """
    install_public_outputs(cls, own_only)
    for name in _FITS:  # glue: wraps the fit method names
        method = cls.__dict__.get(name)
        if isinstance(method, available_if):
            setattr(cls, name, available_if(method.check)(_track_fit(method.fn)))
        elif inspect.isfunction(method):
            setattr(cls, name, _track_fit(method))
    for name in _LOADS:  # glue: wraps the loader method names
        # A full install also wraps an inherited package loader (package
        # classes carry no `_loader` wrapper of their own any more), so an
        # application subclass's `load` still marks the instance fitted.
        method = cls.__dict__.get(name) if own_only else inspect.getattr_static(cls, name, None)
        if isinstance(method, classmethod) and not getattr(method.__func__, "_mojolearn_loader", False):
            setattr(cls, name, _loader(method.__func__))
    if not getattr(cls, "_sklearn_requires_fit", True):
        for name in _NEEDS_FIT:  # glue: unwraps the method names that need a fit
            method = getattr(cls, name, None)
            if getattr(method, "_mojolearn_needs_fit", False):  # guarded on a stateful base
                # a native method (native_method) owns its guard and has no
                # Python wrapper to remove; it stays as it is
                method = getattr(method, "__wrapped__", method)
            if inspect.isfunction(method) and not getattr(method, "_mojolearn_validates", False):
                method = _needs_fit(method, name, require_fit=False)  # validates X, needs no fit
            if method is not None and method is not getattr(cls, name, None):
                setattr(cls, name, method)
        return
    for name in _NEEDS_FIT:  # glue: wraps the method names that need a fit
        if own_only and name not in cls.__dict__:
            continue  # an inherited native method owns its guard
        raw = inspect.getattr_static(cls, name, None)
        if isinstance(raw, available_if):
            if not getattr(raw.fn, "_mojolearn_needs_fit", False):
                setattr(cls, name, available_if(raw.check)(_needs_fit(raw.fn, name)))
            continue
        method = getattr(cls, name, None)
        if (getattr(method, "_mojolearn_validates", False)
                and not getattr(method, "_mojolearn_needs_fit", False)):
            method = method.__wrapped__  # a stateless base's validate-only guard
        if inspect.isfunction(method) and not getattr(method, "_mojolearn_needs_fit", False):
            setattr(cls, name, _needs_fit(method, name))


class available_if:
    """A method that exists only when `check(self)` is true, as
    scikit-learn's `sklearn.utils._available_if.available_if` does it:
    `hasattr(est, name)` is False otherwise, so scikit-learn's checks and
    meta-estimators skip a method this configuration does not have
    (`SVC(probability=False).predict_proba`, `LocalOutlierFactor().predict`).
    `check` returns a bool or raises AttributeError. scikit-learn is not
    needed."""

    def __init__(self, check):
        self.check = check
        self.fn = None

    def __call__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)
        # The name the refusal reports, resolved once here (and by
        # `__set_name__` for a class attribute), not read on every access.
        self.attribute_name = fn.__name__
        return self

    def __set_name__(self, owner, name):
        self.attribute_name = name

    # Per access: the one parameter check and the bound method. The answer
    # depends on a constructor parameter that `set_params` may change
    # (novelty, voting, probability, loss, prediction_data), so it cannot be
    # resolved at class creation.
    def __get__(self, obj, owner=None):
        if obj is None:
            return self.fn
        try:
            ok = self.check(obj)
        except AttributeError as exc:
            raise AttributeError(
                f"This {type(obj).__name__!r} has no attribute {self.attribute_name!r}") from exc
        if not ok:
            raise AttributeError(f"This {type(obj).__name__!r} has no attribute {self.attribute_name!r}")
        return self.fn.__get__(obj, owner)


#: The public methods whose results scikit-learn users index, reduce and
#: compare as NumPy arrays.
_OUTPUTS = ("predict", "predict_proba", "predict_log_proba", "decision_function", "transform",
            "fit_transform", "inverse_transform", "score_samples", "fit_predict", "kneighbors",
            "sample")

_PKG = __name__.rpartition(".")[0]
_PKG_DOT = _PKG + "."


#: The wrapper functions this module puts around estimator methods; their
#: frames are skipped when looking for the real caller.
_WRAPPER_CODES = frozenset(("public", "guarded", "tracked", "load"))


def _caller_outside(frame):
    """True when `frame` (skipping this module's own method wrappers)
    belongs to code outside the package: the caller is a user or
    scikit-learn, not another mojolearn method."""
    while (frame is not None and frame.f_code.co_name in _WRAPPER_CODES
           and frame.f_globals.get("__name__") == __name__):
        frame = frame.f_back
    if frame is None:
        return True
    globs = frame.f_globals
    mod = globs.get("__name__") or ""
    # The identity harness (tools/identity_break.py) is mojolearn's own code
    # under any module name: `__main__` when run directly, and
    # `mojolearn_verify_all_*` when the verifier loads it by path
    # (_verify_all._load_by_path). It marks itself `__mojolearn_internal__` and
    # must see the native results the identity records hash, not the widened
    # public ones, or the harness and the verifier hash different bytes.
    # The native runtime audit stands in for a user's code.
    if globs.get("__mojolearn_user_boundary__") is True:
        return True
    if globs.get("__mojolearn_internal__") is True:
        return False
    return mod != _PKG and not mod.startswith(_PKG_DOT) and not mod.startswith("mojolearn_verify_all_")


def _ndarray(value, np):
    """`value` as a NumPy array when it is an `Array` (a zero-copy view of
    the same memory, `Array.__array__`); the Array members of a tuple, or
    of a list of per-output Arrays, likewise."""
    if type(value) is _Array:
        return value.__array__()
    if type(value) is tuple:
        return tuple(_ndarray(v, np) for v in value)  # glue: converts the members of a result tuple
    if type(value) is list and value and type(value[0]) is _Array:
        return [_ndarray(v, np) for v in value]  # glue: one array per output (multi-output predict_proba)
    return value


def _public_result(est, name, out):
    """A public method's result for a caller outside the package.

    With NumPy loaded (it always is under scikit-learn), an `Array` result
    becomes a `numpy.ndarray` view of the same memory, so the result
    supports everything scikit-learn and its users do with one (fancy
    indexing, `sum(axis=...)`, comparisons, `round`, `dtype.kind`). Without
    NumPy loaded the `Array` is returned unchanged (the NumPy-free
    contract). `predict` keeps the kernel's dtype (`_predict_output`)."""
    np = sys.modules.get("numpy")
    if np is None:
        return out
    # An outlier detector's scores are float64, as scikit-learn's are
    # (`_sk_validate.after_call`). The widening must happen here, on the
    # Array: this wrapper runs inside `_needs_fit`'s guard, so the guard's
    # own `after_call` only ever sees the NumPy view made below.
    out = _sk_validate.after_call(est, name, out)
    return _ndarray(out, np)


def _public_output(method, name):
    if name == "predict":
        # predict has one result for every caller (Andrew, 2026-11-22, lane
        # predict-dtype): the kernel's dtype as a NumPy view when NumPy is
        # loaded, with no caller test and no float64 widening of a
        # regressor's float32 values.
        @functools.wraps(method)
        def public(self, *args, **kwargs):
            out = method(self, *args, **kwargs)
            np = sys.modules.get("numpy")
            return out if np is None else _ndarray(out, np)
    else:
        @functools.wraps(method)
        def public(self, *args, **kwargs):
            out = method(self, *args, **kwargs)
            if _caller_outside(sys._getframe(1)):
                return _public_result(self, name, out)
            return out

    public._mojolearn_public_out = True
    return public


def install_public_outputs(cls, own_only=False):
    """Give `cls`'s public result methods (`_OUTPUTS`) NumPy results for
    callers outside the package (`_public_result`). Calls between the
    library's own estimators keep the `Array`s their code is written for."""
    for name in _OUTPUTS:  # glue: wraps the public result method names
        if own_only and name not in cls.__dict__:
            continue
        raw = inspect.getattr_static(cls, name, None)
        if isinstance(raw, available_if):
            if not getattr(raw.fn, "_mojolearn_public_out", False):
                setattr(cls, name, available_if(raw.check)(_public_output(raw.fn, name)))
            continue
        if inspect.isfunction(raw) and not getattr(raw, "_mojolearn_public_out", False):
            setattr(cls, name, _public_output(raw, name))


class CheckMixin:
    """`estimator.check(X)`: the GPU-versus-CPU check as a call (Andrew,
    2026-11-20), on every estimator through `PublicOutputsMixin` and mixed into
    the estimator families with a base of their own (`CNNClassifier`). A door
    onto `mojolearn.check` (`_verify_check.check`), the function `python -m
    mojolearn verify --check` runs: no second implementation. The check
    modules load inside the method, so `import mojolearn` and every fit,
    predict and transform path are unchanged. Not a constructor parameter,
    so `get_params` and the scikit-learn tags never see it."""

    # No instance state: an empty slot list keeps the instance layout (and the
    # native attribute layouts that read the first `__dict__` descriptor in
    # the MRO) exactly what it was before this base was added.
    __slots__ = ()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # glue: once per class definition, the class's rows of the digest
        # field tables (_digest_fields.py); nothing per instance or per fit.
        _digest_declare(cls)

    @property
    def digest_(self):
        """The hash of this fitted model, as one lower-case hex string.

        In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA
        and AMD GPUs, and the CPU binaries are held to the same bits. So the
        same estimator fitted on the same data has the same `digest_` on any
        of those machines and with ``device="cpu"``, within one Mojolearn
        version: fit twice, compare the two strings, and any difference means
        the data, the parameters or the version changed.

        The digest means "this model gives these answers". It covers
        exactly the state that decides what `predict`, `predict_proba`,
        `transform`, `score` and the other answer methods return: the class
        name, the architecture, the parameters those methods read, and every
        fitted array and value (weights, buffers such as batch-norm running
        statistics, class labels, category maps), by name, type, shape and
        bytes. It leaves out how the model was trained: optimizer moments,
        iteration, epoch and step counters, loss and score histories,
        convergence records, timing, the device and the thread count (see
        `training_digest_`). So a model, its saved, exported, pickled and
        loaded copies, and the same model on another device have one
        digest, and two models with different digests give different
        answers or were hashed from different state. It is computed each
        time it is read, from the model as it is at that moment, and never
        during `fit`; calling `predict`, `transform` or `score` does not
        change it. A meta-estimator's digest includes the digests of its
        fitted members. For `IterativeImputer` with
        ``sample_posterior=True``, `LatentDirichletAllocation` and
        `MinCovDet`, whose random draws continue from one call to the next
        and so decide the next answers, the digest changes as the model is
        used.

        Raises
        ------
        AttributeError
            If the model is not fitted, if it runs in `fast` mode (Apple
            only), which promises no identical bits, or if the model holds a
            fitted value that cannot be hashed.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
        >>> a = mojolearn.Ridge(alpha=1.0).fit(X, X[:, 0])
        >>> b = mojolearn.Ridge(alpha=1.0, device="cpu").fit(X, X[:, 0])
        >>> a.digest_ == b.digest_    # doctest: +SKIP
        True
        """
        # One native call (bindings/fitted_digest_door.mojo); nothing is
        # hashed or looked up at fit.
        return _digest_read(self)

    @property
    def training_digest_(self):
        """The hash of how this model was trained, as one lower-case hex
        string, separate from `digest_`.

        It covers the training record `digest_` leaves out, where the model
        keeps one: the optimizer's moments and step counters a resumed fit
        reads, the iteration and epoch counts, the loss and score histories
        and the convergence records. Two fits with the same data, parameters
        and seed have the same `training_digest_` on every machine, within
        one Mojolearn version. A model loaded from an inference export,
        which drops the training record, keeps its `digest_` and has a
        different `training_digest_`. It is computed each time it is read.

        Raises
        ------
        AttributeError
            If the model is not fitted or runs in `fast` mode (Apple only).

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> X = np.random.default_rng(0).random((200, 4), dtype=np.float32)
        >>> a = mojolearn.SGDRegressor(random_state=0).fit(X, X[:, 0])
        >>> b = mojolearn.SGDRegressor(random_state=0).fit(X, X[:, 0])
        >>> a.training_digest_ == b.training_digest_    # doctest: +SKIP
        True
        """
        # The same native call as `digest_`, with the training selector.
        return _digest_read(self, _DIGEST_TRAINING_ARGUMENTS)

    def check(self, X, *, method=None, fit=None):
        """Run this fitted model on the GPU and on the CPU, print both sides'
        reports and whether their answers are the same bits.

        In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA
        and AMD GPUs, and the CPU binaries are held to the same bits. This
        answers `X` with this model twice on this machine, in this process: on
        the GPU, and on the CPU from the same fitted model on `device="cpu"`.
        It runs in `identical` mode only.

        It prints, for each side, the device and the binaries that answered,
        what the runtime counted while the side ran (launches, uploads,
        downloads), the wall time, and the shape, dtype, SHA-256 and first
        values of each output; then one line: MATCH, or MISMATCH with the
        index and the two values of the first differing element. When nothing
        could be compared (the estimator is not in `identical` mode, or the
        class does not run on the CPU) it prints why. Without a GPU it prints
        the CPU side and says the GPU side is not applicable.

        Parameters
        ----------
        X : array-like
            The rows to answer for.
        method : str, optional
            The method to check, for example `'predict'`, `'predict_proba'`
            or `'transform'`. Default: `predict` when the estimator has it,
            else `transform`.
        fit : tuple, optional
            Arguments for `fit`. When given, the estimator is fitted on the
            GPU first, then the fitted model is checked on both sides. Fitting
            is never done unless asked for.

        Returns
        -------
        dict
            The same result as `mojolearn.check(estimator, X)`: `ran`,
            `sides` (one record per side) and `verdict["verdict"]`, which is
            `'MATCH'` or `'MISMATCH'`; `ran` is False, with a `reason`, when
            nothing was compared.

        Examples
        --------
        >>> import numpy as np, mojolearn
        >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
        >>> model = mojolearn.RandomForestRegressor(n_estimators=20).fit(X, X[:, 0])
        >>> result = model.check(X)    # doctest: +SKIP
        >>> result["verdict"]["verdict"]    # doctest: +SKIP
        'MATCH'
        """
        # Loaded here, on the first call only: nothing of the check is
        # imported with the package or touched by fit/predict/transform.
        from ._verify_check import check, format_check
        result = check(self, X, method=method, fit=fit)
        print(format_check(result))
        return result


# glue: the one `digest_` property object, for the native attribute lookups
# that admit it by identity (bindings/fitted_digest_property_pack.mojo and the
# property recipe of _native_public_attributes.connect_classes).
_DIGEST_CONNECTION["property"] = vars(CheckMixin)["digest_"]
_DIGEST_CONNECTION["training_property"] = vars(CheckMixin)["training_digest_"]


class PublicOutputsMixin(CheckMixin):
    """NumPy results for callers outside the package: the public result
    methods (`install_public_outputs`) and the fitted attributes. A public
    attribute ending in an underscore (`coef_`, `labels_`,
    `cluster_centers_`) that holds an `Array` reads as a `numpy.ndarray`
    view of the same memory when NumPy is loaded and the reader is not
    mojolearn's own code; the instance keeps the `Array`.

    Every estimator also gets `check(X)` and `digest_` from here
    (`CheckMixin`)."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # Resolved once, at class creation: a package class's public methods
        # are one-call native doors whose packs return the public result
        # (bindings/classical_caller_unpack.mojo reads the caller natively), so
        # they get no Python `public` wrapper and no per-call frame walk. Before
        # 2026-11-15 a package class not yet marked native was wrapped here and
        # the trees bridges unwrapped it again after registration.
        if getattr(cls, "_mojolearn_native_public", False) or not _application_subclass(cls):
            return
        install_public_outputs(cls)


class SklearnTagsMixin(PublicOutputsMixin):
    """`__sklearn_tags__` for an estimator: see `sklearn_tags`. Subclasses
    get fit tracking and `NotFittedError` (see `install_fit_tracking`)."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # A package class: the native owner tracks fits, validates inputs and
        # refuses an unfitted call itself; no `tracked`/`guarded`/`load`
        # wrapper is installed. An application subclass's own overrides are
        # still guarded (all inherited methods too when it derives from a class
        # not marked native).
        if not _application_subclass(cls):
            return
        install_fit_tracking(cls, own_only=bool(getattr(cls, "_mojolearn_native_public", False)))

    def __sklearn_tags__(self):
        return sklearn_tags(self)

    def __sklearn_is_fitted__(self):
        return _MODE_METADATA(self, "is_fitted", _MODE_METADATA_CONTEXT)


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

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _connect_application_params(cls)

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
        return _PARAMS_METADATA(self, deep, _PARAMS_METADATA_CONTEXT)

    def set_params(self, **params):
        """Set constructor parameters. The estimator is rebuilt through its
        constructor, so every constructor check runs again and any fitted
        state is dropped, as a new setting invalidates it.

        Parameters
        ----------
        **params : keyword arguments
            Constructor parameters; use ``name__parameter`` for nested values.

        Returns
        -------
        self : object
            This estimator with its updated parameters.
        """
        # One native call (bindings/full_reset_params_native_door.mojo): connected
        # native schemas first, else the generic rebuild (_generic_reset).
        return _GENERIC_PARAMS(self, params, _GENERIC_PARAMS_CONTEXT)


def _connect_application_params(cls):
    # glue: an application subclass's parameter schema, built once when its
    # class statement runs (see _native_params_metadata.connect_subclass)
    context = globals().get("_PARAMS_METADATA_CONTEXT")
    if context is not None and _application_subclass(cls):
        from ._native_params_metadata import connect_subclass
        connect_subclass(context, cls)


def _install_numeric_mode(cls, *, native_constructor=False):
    # glue: NumericModeMixin.__init_subclass__'s class-definition wrapper
    # One row of the device table per estimator class (which classes run
    # device="cpu" beside a GPU set; generated, _device_table.py).
    _device_record(cls)
    if getattr(cls, "_mojolearn_native_public", False):
        return  # the constructor takes numeric_mode itself
    # The former CPU-training guard loop over the five fit names is deleted
    # (lane py-mode-wrappers, 2026-11-15): `_guard_cpu_training` returns the
    # method unchanged, so the loop set each fit method to itself.
    orig = cls.__dict__.get("__init__")
    if orig is None:
        return
    if native_constructor:
        # A class-creation request, not an inherited native-public marker.
        # These constructors already hand the explicit mode to their native
        # owner; preserve the original function instead of wrapping/unwrapping.
        if "numeric_mode" not in inspect.signature(orig).parameters:
            raise TypeError("native constructor must explicitly accept numeric_mode")
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
    # `device=` joins this wrapper when the package connects the class
    # (`install_device`), once the compiled-constructor installers have
    # taken the classes that are theirs.
    _DEVICE_PENDING.append((cls, __init__, orig))


#: (class, its numeric-mode wrapper, its own constructor): the classes whose
#: wrapper may take `device=`, decided by `install_device`.
_DEVICE_PENDING = []


def install_device():
    """Give `device=None | "gpu" | "cpu"` to every class whose constructor is
    still the shared numeric-mode wrapper, in the one place that wrapper is
    made (lane dw-core, 2026-11-20).

    Called by the package's connection pass (`_families._connect_pass`) after
    the compiled-constructor installers and before any constructor identity
    is recorded. A class takes the keyword here when ALL of this holds, so
    that the parameter is carried by code that already exists:
      * its constructor is still this module's wrapper (no installer replaced
        it with a compiled constructor, whose argument schema is fixed);
      * its parameters are its constructor signature (`ParamsMixin.get_params`)
        and `set_params` is the generic rebuild through the constructor
        (`ParamsMixin.set_params`): get_params, set_params, clone and pickle
        then carry `device`, and `set_params` runs the constructor's check
        before the estimator changes;
      * it has no compiled constructor recipe (`_connect_native_init`, the
        mode-reconstruction recipes), whose argument list is fixed;
      * its own constructor does not already name a `device`.
    Every other class names `device` in its own compiled constructor schema
    (its family's work); it is listed by `device_report()`.

    The value is checked in compiled code where it is given
    (bindings/mode_metadata_native_door.mojo, operation `device_admit`):
    an unknown value, 'gpu' on a CPU-only installation and 'cpu' for a class
    that cannot run there beside a GPU set (`_device_table.CPU_CLASSES`) are
    refused in words before the constructor body runs."""
    from ._native_mode_reconstruction import _NATIVE_RECIPES
    pending, _DEVICE_PENDING[:] = list(_DEVICE_PENDING), []
    native_inits = _GENERIC_PARAMS_CONTEXT["native_inits"]
    package = __name__.rsplit(".", 1)[0] + "."
    for cls, wrapper, orig in pending:  # glue: class connection, once per class, never product execution
        if _application_subclass(cls):
            # A caller's own subclass keeps the constructor it wrote; its
            # parameter schema was built from that signature at its class
            # statement (`_connect_application_params`).
            _DEVICE_SKIPPED[cls] = "an application subclass"
            continue
        if cls.__dict__.get("__init__") is not wrapper:
            _DEVICE_SKIPPED[cls] = "its constructor is compiled"
            continue
        if (getattr(cls, "get_params", None) is not ParamsMixin.get_params
                or getattr(cls, "set_params", None) is not ParamsMixin.set_params):
            _DEVICE_SKIPPED[cls] = "its parameters are a declared list"
            continue
        if cls in native_inits or any(  # glue: fixed recipe owners in the class's ancestry
                base.__module__.removeprefix(package) + "." + base.__name__ in _NATIVE_RECIPES
                for base in cls.__mro__):
            _DEVICE_SKIPPED[cls] = "its constructor has a compiled recipe"
            continue
        sig = inspect.signature(wrapper)
        if "device" in sig.parameters:
            _DEVICE_SKIPPED[cls] = "its own constructor names device"
            continue
        cls.__init__ = _device_wrapper(wrapper, orig, sig)
        _DEVICE_INSTALLED.append(cls)


#: The classes `install_device` gave the keyword, and for each class it did
#: not, why (`device_report`).
_DEVICE_INSTALLED = []
_DEVICE_SKIPPED = {}
_OBJECT_GET = object.__getattribute__


def _device_wrapper(wrapper, orig, sig):
    # glue: the numeric-mode wrapper of `_install_numeric_mode`, with `device`
    @functools.wraps(orig)
    def __init__(self, *args, numeric_mode=None, device=None, **kwargs):
        # `device` is checked in compiled code where it is given, before the
        # constructor body runs. The default None on a new instance calls
        # nothing; a constructor run again on a live instance drops its
        # route. The instance dictionary is read through object's own lookup:
        # two membership tests, no attribute door and no AttributeError built.
        fields = _OBJECT_GET(self, "__dict__")
        again = "device" in fields
        if device is not None or again:
            _MODE_METADATA(self, "device_admit", _MODE_METADATA_CONTEXT, device)
        orig(self, *args, **kwargs)
        if numeric_mode is not None or not hasattr(self, "numeric_mode"):
            self.numeric_mode = numeric_mode
        # A base class's wrapper runs inside `orig` and stores its own None
        # first; the outermost wrapper stores last. A constructor run again
        # stores what it was given (None too).
        if device is not None or again or "device" not in fields:
            self.device = device

    params = list(sig.parameters.values())
    at = next((i for i, p in enumerate(params) if p.kind is p.VAR_KEYWORD), len(params))  # glue: signature parameter position
    params.insert(at, inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY, default=None))
    __init__.__signature__ = sig.replace(parameters=params)
    return __init__


def device_report():
    """One row per estimator class of the loaded families: `(class name,
    takes device=, the device="cpu" answer beside a GPU set)`. The answer is
    True, or the sentence the constructor's refusal carries; for a class that
    does not take the keyword through the shared wrapper the second field is
    why. Sorted by class name; for reports and tests, never product code."""
    rows = []
    for cls, answer in _DEVICE_CPU_CLASSES.items():  # glue: report over the class table
        takes = (cls in _DEVICE_INSTALLED or "device" in inspect.signature(cls.__init__).parameters
                 if cls.__init__ is not object.__init__ else False)
        rows.append((cls.__module__ + "." + cls.__name__, takes if takes else _DEVICE_SKIPPED.get(cls, False), answer))
    return sorted(rows, key=lambda row: row[0])  # glue: report order


class NumericModeMixin(ParamsMixin):
    """Gives an estimator `self._bind(name)`.

    `numeric_mode` is read off the instance at every call rather than
    resolved in `__init__`, so an estimator that is unpickled from an older
    version (no such attribute) falls back to the process default instead of
    raising -- and so that setting the attribute after construction works the
    way a caller would expect it to.
    """

    def __init_subclass__(cls, *, _native_constructor=False, **kw):
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
        # This keyword is consumed only for this class definition. A user's
        # subclass with its own constructor still gets ordinary mode injection.
        super().__init_subclass__(**kw)
        _install_numeric_mode(cls, native_constructor=_native_constructor)
        _connect_application_params(cls)

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
            ``'identical'`` or ``'fast'`` (Apple only), read from the compiled
            binary the estimator uses, not from the argument it was given.
        """
        return _MODE_METADATA(self, "numeric_mode_used", _MODE_METADATA_CONTEXT)

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
        return _MODE_METADATA(self, "vendor_used", _MODE_METADATA_CONTEXT)


# Import-time ABI connection precedes every estimator subclass definition.
# The final package connection supplies exact class/binary routes later, without
# replacing any method identities captured by other native owner layouts.
from ._native_mode_metadata import connect as _connect_mode_metadata
_MODE_METADATA, _MODE_METADATA_CONTEXT = _connect_mode_metadata()
# The one table the compiled `device_admit` check reads: class -> True, or
# why device="cpu" is refused for it (_device_table.py). The same dictionary
# object: classes defined later add their row.
_MODE_METADATA_CONTEXT["device_cpu"] = _DEVICE_CPU_CLASSES
# The `digest_` connection (the property object, the compiled entry and its
# context; _digest_fields.py), for the classical layouts' native lookup.
_MODE_METADATA_CONTEXT["digest_connection"] = _DIGEST_CONNECTION
# The lazy CPU-image connection the same check calls on an admitted
# device="cpu" beside a GPU set (_device_table.connect_cpu).
from ._device_table import connect_cpu as _device_connect_cpu
_MODE_METADATA_CONTEXT["device_cpu_connect"] = _device_connect_cpu
from ._native_loaded_modules import connect_module as _connect_native_module
# GP lifecycle admission reads this initialized owner without importing or
# invoking Python. Later class connections extend the same metadata dictionary.
_connect_native_module(__name__, sys.modules[__name__], ('_MODE_METADATA_CONTEXT',))
from ._native_params_metadata import connect as _connect_params_metadata
_PARAMS_METADATA, _PARAMS_METADATA_CONTEXT = _connect_params_metadata()
# The generic public attribute and parameter-setter slots of every class
# without its own connected native slot. The setter context is the full-reset
# owner's context (_native_full_reset_params._CONTEXT), connected later.
from ._native_mode_metadata import connect_generic_attributes as _connect_generic_attributes
_GENERIC_ATTRIBUTE, _GENERIC_ATTRIBUTE_CONTEXT = _connect_generic_attributes()
# Import-time connection: ordinary reads enter the native descriptor directly,
# including the fallback classes not yet admitted to specialized native plans.
PublicOutputsMixin.__getattribute__ = _GENERIC_ATTRIBUTE
from ._native_mode_metadata import connect_generic_params as _connect_generic_params
_GENERIC_PARAMS, _GENERIC_PARAMS_CONTEXT = _connect_generic_params()


def _connect_native_init(cls, entry, operation, names, owner):
    # glue: import-time registration, once per class, of a class whose public
    # __init__ body is exactly `entry(self, operation, (<names>...), owner)`.
    # The generic set_params rebuild (bindings/full_reset_params_native_door.mojo
    # `_generic_reset`) then calls that native constructor directly instead of
    # re-entering the Python __init__ body. `names` is the argument tuple in the
    # body's order; None stands for a fixed None argument. Keep it in step with
    # the __init__ body beside it.
    if cls.__new__ is not object.__new__:
        raise TypeError('native constructor registration requires object.__new__')
    declared = tuple(inspect.signature(cls.__init__).parameters)[1:]
    if sorted(name for name in names if name is not None) != sorted(declared):  # glue: import-time schema check
        raise TypeError(cls.__name__ + ': native constructor arguments differ from its signature')
    _GENERIC_PARAMS_CONTEXT['native_inits'][cls] = (entry, operation, tuple(names), owner, object.__new__)
