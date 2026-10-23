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
    hw = {}
    names = ("holtwinters_fit", "holtwinters_forecast", "holtwinters_predict")
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            module = _backend.binding("_mojolearn_tsa", mode)
            symbols = {}
            for name in names:  # glue: fixed compiled entry names
                fn = getattr(module, name, None)
                if isinstance(fn, BuiltinFunctionType):
                    symbols[name] = fn
            hw[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            hw[mode] = type(error).__name__ + ": " + str(error)
    context["hw_bindings"] = hw
    context["hw_entry_names"] = names
    # the host inference class reads its loaded forecast binding from the
    # module table at call time (its own load loads it)
    context["hw_host_module"] = "mojolearn._host._mojolearn_forecast_host"
    return entry, context
