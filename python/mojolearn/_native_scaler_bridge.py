# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for StandardScaler's native fit owner."""


def connect():
    # glue: retain compiled ABI entries and buffer metadata once at import
    from types import BuiltinFunctionType, ModuleType
    from . import _backend
    from ._native_classical_bridge import _numpy_free_context
    from .tokenizer import _binding

    context = _numpy_free_context()
    context["native_module_type"] = ModuleType
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI connection
        try:
            prep = _backend.binding("_mojolearn_preprocessing", mode)
            xprep = _backend.binding("_mojolearn_x_prep", mode)
            entries = {}
            for name in ("standard_fit", "standard_fit_direct", "preprocessing_numeric_mode"):  # glue: cache compiled scaler symbols
                try:
                    entry = getattr(prep, name)
                except (AttributeError, ImportError):
                    if name == "standard_fit_direct":
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"StandardScaler requires compiled {name}")
                entries[name] = entry
            for name in ("x_prep_run", "x_prep_numeric_mode"):  # glue: cache the existing native weighted-statistics executor
                entry = getattr(xprep, name)
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"StandardScaler requires compiled {name}")
                entries[name] = entry
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    context["scaler_bindings"] = connected
    entry = _binding()._module.standard_scaler_fit_native
    if not isinstance(entry, BuiltinFunctionType):
        raise TypeError("StandardScaler requires compiled standard_scaler_fit_native")
    return entry, context
