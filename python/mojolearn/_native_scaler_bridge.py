# SPDX-License-Identifier: Apache-2.0
"""Import-time ABI connection for StandardScaler's native fit owner."""


def _entries(prep, xprep):
    # glue: the compiled scaler and x_prep exports of one pair of images, connected during import
    from types import BuiltinFunctionType
    from . import _backend
    # The provider admission (providers/python/neural_descriptor_pyapi_door.mojo
    # `neural_descriptor_entry`) takes an exact module that owns its
    # entry. On a CPU-only install `_backend.binding` returns the lazy
    # `_HostBinding` facade; resolve it to its host module here, as
    # _native_family_bridge.connect_bindings does, or GP normalize_y
    # refused "native provider requires an exact compiled module"
    # (smoke-triage 2026-11-15: lanes gp-normalize-y, gp-sample-y-normalize).
    entries = {"native_scaler_provider": _backend.exact_host_module(prep),
               "native_program_provider": _backend.exact_host_module(xprep)}
    for name in ("standard_fit", "standard_fit_direct", "standard_transform", "standard_transform_direct", "standard_fit_transform_direct", "standard_fit_transform", "preprocessing_numeric_mode"):  # glue: cache compiled scaler symbols
        try:
            entry = getattr(prep, name)
        except (AttributeError, ImportError):
            if name in ("standard_fit_direct", "standard_transform_direct", "standard_fit_transform_direct", "standard_fit_transform"):
                continue
            raise
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"StandardScaler requires compiled {name}")
        entries[name] = entry
    for name in ("x_prep_run", "x_prep_numeric_mode", "x_prep_vendor"):  # glue: cache the existing native weighted-statistics executor
        entry = getattr(xprep, name)
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"StandardScaler requires compiled {name}")
        entries[name] = entry
    return entries


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
            connected[mode] = _entries(_backend.binding("_mojolearn_preprocessing", mode),
                                       _backend.binding("_mojolearn_x_prep", mode))
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
    # The CPU context of `device="cpu"`: the host preprocessing and x_prep images beside the device ones.
    from ._native_prep_device import connect_cpu_device
    connect_cpu_device(context, lambda load: _entries(load("_mojolearn_preprocessing_host"),
                                                      load("_mojolearn_x_prep_host")))
    return entry, call, context


def register_subclass(cls, context):
    # glue: register an ordinary class layout at class definition time
    from ._native_classical_bridge import _layout
    context["classical_public_layouts"][cls] = _layout(cls, context["classical_attribute_descriptor"])
    for base in type.__dict__["__mro__"].__get__(cls)[1:]:  # glue: preserve explicit CPU scope through ordinary subclasses
        if base in context["scaler_host_contexts"]:
            context["scaler_host_contexts"][cls] = context["scaler_host_contexts"][base]
            break


def connect_host(cls, context, prep, xprep, training):
    # glue: called only by _classical_host at its explicit CPU API setup
    from types import BuiltinFunctionType
    host = dict(context)
    entries = {"standard_transform": prep.standard_transform,
               "preprocessing_numeric_mode": prep.estimators_numeric_mode,
               "x_prep_run": xprep.x_prep_run,
               "x_prep_numeric_mode": xprep.x_prep_numeric_mode,
               "x_prep_vendor": xprep.x_prep_vendor}
    if any(not isinstance(entry, BuiltinFunctionType) for entry in entries.values()):  # glue: host ABI identity
        raise TypeError("HostStandardScaler requires compiled CPU entries")
    entries["native_program_provider"] = xprep
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
            entries["native_scaler_provider"] = training
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
