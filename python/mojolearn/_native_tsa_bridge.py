# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the stationarity functions
(`kpss_test`, `select_d`; bindings/tsa_public_native.mojo). Python here
only connects the compiled `_mojolearn_tsa` entries of both tiers (the lane
is identical only: a tier without a binary keeps its refusal, raised by the
call that asks for it) and the shared context, once, at import."""

_ENTRIES = ("kpss_test", "select_d")


def connect():
    """`(entry, context)` for `kpss_test` and `select_d`."""
    from types import BuiltinFunctionType
    from . import _backend
    from .tokenizer import _binding
    from ._native_classical_bridge import _state_context
    entry = _binding()._module.tsa_native_call
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("the stationarity functions require the compiled tsa_native_call entry")
    context = _state_context()
    selected = _backend.default_mode()
    bindings = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            module = _backend.binding("_mojolearn_tsa", mode)
            symbols = {}
            for name in _ENTRIES:  # glue: fixed compiled entry names
                fn = getattr(module, name)
                if not isinstance(fn, BuiltinFunctionType):
                    raise TypeError(f"the stationarity functions require a compiled {name}")
                symbols[name] = fn
            bindings[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            bindings[mode] = type(error).__name__ + ": " + str(error)
    context["tsa_bindings"] = bindings
    return entry, context
