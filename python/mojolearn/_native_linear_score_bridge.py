# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native `score` of the linear expansion
regressors (bindings/linear_score_native.mojo): predict and R^2 in one call
into the compiled classical owner. Python here only retains the compiled
entries, once, at import."""


def _entries(binding, names):
    # glue: retain checked compiled entries of one binding at connection
    from types import BuiltinFunctionType
    entries = {}
    for name in names:  # glue: fixed compiled entry names
        entry = getattr(binding, name)
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"linear score requires compiled {name}")
        entries[name] = entry
    return entries


def connect():
    """`(entry, context)` for `_LinearRegressorMixin.score`."""
    from types import BuiltinFunctionType
    from . import _backend
    from ._native_classical_bridge import _state_context
    try:
        from .tokenizer import _binding
        entry = _binding()._module.classical_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("linear score requires the compiled classical_native_call entry")
        context = _state_context()
    except ImportError as error:
        message = f"mojolearn: score needs the compiled host binding: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"linear_score_unavailable": message}
    selected = _backend.default_mode()
    linear, metrics = {}, {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            linear[mode] = _entries(_backend.binding("_mojolearn_x_linear", mode), ("x_linear_decision",))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            linear[mode] = type(error).__name__ + ": " + str(error)
        try:
            metrics[mode] = _entries(_backend.binding("_mojolearn_metrics", mode), ("r2_score",))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            metrics[mode] = type(error).__name__ + ": " + str(error)
    context.update(linear_score_bindings=linear, linear_score_metrics=metrics)
    return entry, context
