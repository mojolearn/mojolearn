# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for StandardScaler's native fit owner."""


def connect(scaler_class):
    # glue: retain compiled ABI entries and buffer metadata once at import
    from types import BuiltinFunctionType, ModuleType, MethodType
    from . import _backend
    from ._native_classical_bridge import _numpy_free_context, _state_context, _layout
    from .tokenizer import _binding

    context = _state_context()
    context["native_module_type"] = ModuleType
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI connection
        try:
            prep = _backend.binding("_mojolearn_preprocessing", mode)
            xprep = _backend.binding("_mojolearn_x_prep", mode)
            entries = {}
            for name in ("standard_fit", "standard_fit_direct", "standard_transform", "standard_transform_direct", "standard_fit_transform_direct", "preprocessing_numeric_mode"):  # glue: cache compiled scaler symbols
                try:
                    entry = getattr(prep, name)
                except (AttributeError, ImportError):
                    if name in ("standard_fit_direct", "standard_transform_direct", "standard_fit_transform_direct"):
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
    call = _binding()._module.standard_scaler_native_call
    if not isinstance(call, BuiltinFunctionType):
        raise TypeError("StandardScaler requires compiled standard_scaler_native_call")
    import sys
    context["classical_getframe"] = sys._getframe
    context["scaler_class"] = scaler_class
    context["scaler_host_contexts"] = {}
    attribute = _binding()._module.standard_scaler_attribute_call
    descriptor = _binding()._module.classical_attribute_descriptor(MethodType(attribute, context))
    scaler_class.__getattribute__ = descriptor
    context.update(classical_attribute_descriptor=descriptor, classical_public_layouts={},
                   classical_public_installed=True)
    context["classical_public_layouts"][scaler_class] = _layout(scaler_class, descriptor)
    return entry, call, context


def register_subclass(cls, context):
    # glue: register an ordinary class layout at class definition time
    from ._native_classical_bridge import _layout
    context["classical_public_layouts"][cls] = _layout(cls, context["classical_attribute_descriptor"])


def connect_host(cls, context, prep, xprep, training):
    # glue: called only by _classical_host at its explicit CPU API setup
    from types import BuiltinFunctionType
    host = dict(context)
    entries = {"standard_transform": prep.standard_transform,
               "preprocessing_numeric_mode": prep.estimators_numeric_mode,
               "x_prep_run": xprep.x_prep_run,
               "x_prep_numeric_mode": xprep.x_prep_numeric_mode}
    if any(not isinstance(entry, BuiltinFunctionType) for entry in entries.values()):  # glue: host ABI identity
        raise TypeError("HostStandardScaler requires compiled CPU entries")
    if isinstance(training, str):
        entries["scaler_training_error"] = training
    else:
        try:
            entry = training.standard_fit
            if not isinstance(entry, BuiltinFunctionType):
                raise TypeError("HostStandardScaler requires compiled standard_fit")
            if training.preprocessing_numeric_mode() != 1 or training.preprocessing_vendor() != "cpu":
                raise ValueError("HostStandardScaler training requires the CPU identical binding")
            entries["standard_fit"] = entry
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            entries["scaler_training_error"] = type(error).__name__ + ": " + str(error)

    host.update(scaler_explicit_host=True, scaler_bindings={"identical": entries})
    context["scaler_host_contexts"][cls] = host


def connect_host_unavailable(cls, context, error):
    # glue: preserve a missing optional host binary as an import-time ABI result
    host = dict(context)
    host.update(scaler_explicit_host=True,
                scaler_bindings={"identical": type(error).__name__ + ": " + str(error)})
    context["scaler_host_contexts"][cls] = host
