# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of MinMaxScaler's compiled owner and fixed ABI."""


def connect(scaler_class):
    # glue: retain compiled ABI entries and metadata during package setup
    from types import BuiltinFunctionType, MethodType
    import sys
    from . import _backend
    from ._native_classical_bridge import _numpy_free_context, _state_context, _layout
    from .tokenizer import _binding

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            prep = _backend.binding("_mojolearn_preprocessing", mode)
            xprep = _backend.binding("_mojolearn_x_prep", mode)
            entries = {}
            for name in ("minmax_fit", "minmax_fit_direct", "minmax_transform", "minmax_transform_direct", "preprocessing_numeric_mode"):  # glue: compiled preprocessing exports
                try:
                    entry = getattr(prep, name)
                except (AttributeError, ImportError):
                    if name in ("minmax_fit_direct", "minmax_transform_direct"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"MinMaxScaler requires compiled {name}")
                entries[name] = entry
            for name in ("x_prep_run", "x_prep_numeric_mode"):  # glue: native missing-value and running-extrema executor
                entry = getattr(xprep, name)
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"MinMaxScaler requires compiled {name}")
                entries[name] = entry
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    module = _binding()._module
    call = module.minmax_scaler_native_call
    attribute = module.standard_scaler_attribute_call
    if not isinstance(call, BuiltinFunctionType) or not isinstance(attribute, BuiltinFunctionType):
        raise TypeError("MinMaxScaler requires compiled native lifecycle and attribute entries")
    context.update(scaler_bindings=connected, scaler_class=scaler_class, scaler_host_contexts={},
                   classical_getframe=sys._getframe)
    descriptor = module.classical_attribute_descriptor(MethodType(attribute, context))
    scaler_class.__getattribute__ = descriptor
    context.update(classical_attribute_descriptor=descriptor, classical_public_layouts={},
                   classical_public_installed=True)
    context["classical_public_layouts"][scaler_class] = _layout(scaler_class, descriptor)
    return call, context


def connect_host(cls, context, prep, xprep):
    # glue: only _classical_host supplies the explicit CPU API's fixed entries
    from types import BuiltinFunctionType
    host = dict(context)
    entries = {"minmax_transform": prep.minmax_transform,
               "preprocessing_numeric_mode": prep.estimators_numeric_mode,
               "x_prep_run": xprep.x_prep_run,
               "x_prep_numeric_mode": xprep.x_prep_numeric_mode}
    if any(not isinstance(entry, BuiltinFunctionType) for entry in entries.values()):  # glue: host ABI identity
        raise TypeError("HostMinMaxScaler requires compiled CPU entries")
    host.update(scaler_explicit_host=True, scaler_bindings={"identical": entries})
    context["scaler_host_contexts"][cls] = host
