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
    from ._native_owner_lifetime import connect_owner_context
    ownership = dict(context)
    connect_owner_context(ownership)
    context["tsa_owner_context"] = ownership
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
    arima = {}
    arima_names = ("arima_fit", "arima_predict", "arima_forecast", "arima_ic_from_loglike", "arima_numeric_mode",
                   "arima_fast_search_mode", "arima_order_caps", "arima_ndiffs_kpss",
                   "arima_order_batch_enabled", "arima_order_search_device", "arima_order_search",
                   "arima_search_reuse_enabled", "arima_order_search_fit")
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes, connected once
        try:
            module = _backend.binding("_mojolearn_arima", mode)
            symbols = {"module_name": module.__name__}
            for name in arima_names:  # glue: fixed compiled entry names
                fn = getattr(module, name, None)
                if isinstance(fn, BuiltinFunctionType):
                    symbols[name] = fn
            arima[mode] = symbols
        except (ImportError, AttributeError, ValueError, RuntimeError, TypeError, OSError) as error:
            arima[mode] = type(error).__name__ + ": " + str(error)
    context["arima_bindings"] = arima
    context["arima_entry_names"] = arima_names
    from ._buffer import _native
    for name in ("gather_rows_bytes", "scatter_rows_bytes", "cast_elements", "bincount_i64",  # glue: fixed compiled registration names
                 "select_fold_i64", "ic_running_min_f32", "ic_running_min_f64", "strided_copy_bytes"):  # glue: fixed compiled row moves, integer casts and IC/group helpers
        fn = _native(name)
        if not isinstance(fn, BuiltinFunctionType):
            raise TypeError(f"AutoARIMA requires the compiled {name} entry")
        context["base_" + name] = fn
    return entry, context
