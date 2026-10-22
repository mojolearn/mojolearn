# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of MaxAbsScaler's native owner and fixed ABI."""


def connect(scaler_class):
    # glue: compiled entries and plain metadata are retained during import
    import sys
    from types import BuiltinFunctionType, MethodType
    from . import _backend
    from ._arena_io import DeviceRows
    from ._native_classical_bridge import _state_context, _numpy_free_context, _layout
    from .tokenizer import _binding

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_x_prep", mode)
            entries = {}
            for name in ("x_prep_run", "x_prep_numeric_mode", "x_prep_maxabs_fit_direct"):  # glue: native exports
                try:
                    entry = getattr(module, name)
                except (AttributeError, ImportError):
                    if name == "x_prep_maxabs_fit_direct":
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"MaxAbsScaler requires compiled {name}")
                entries[name] = entry
            entries["preprocessing_numeric_mode"] = entries["x_prep_numeric_mode"]
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    base = _backend.binding("_mojolearn", "identical")
    gather, finite32, finite64 = base.gather_rows_bytes, base.all_finite_f32, base.all_finite_f64
    door = _binding()._module
    call, attribute = door.maxabs_scaler_native_call, door.standard_scaler_attribute_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, attribute, gather, finite32, finite64)):  # glue: fixed native ABI identity
        raise TypeError("MaxAbsScaler requires compiled lifecycle, attribute and row-gather entries")
    context.update(scaler_bindings=connected, scaler_class=scaler_class, scaler_host_contexts={},
                   maxabs_rows_class=DeviceRows, maxabs_gather_rows=gather,
                   maxabs_finite_f32=finite32, maxabs_finite_f64=finite64, classical_getframe=sys._getframe)
    descriptor = door.classical_attribute_descriptor(MethodType(attribute, context))
    scaler_class.__getattribute__ = descriptor
    context.update(classical_attribute_descriptor=descriptor, classical_public_layouts={},
                   classical_public_installed=True)
    context["classical_public_layouts"][scaler_class] = _layout(scaler_class, descriptor)
    return call, context
