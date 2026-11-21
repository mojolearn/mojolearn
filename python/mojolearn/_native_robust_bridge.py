# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of RobustScaler's native owner and fixed ABI."""


def _entries(module):
    # glue: the fixed x_prep exports of one compiled image, connected during import
    from types import BuiltinFunctionType
    entries = {}
    for name in ("x_prep_run", "x_prep_numeric_mode", "x_prep_vendor", "x_prep_host_column",  # glue: fixed native exports connected during import
                 "x_prep_run_out", "x_prep_run_scratch", "x_prep_run_ranges",
                 "x_prep_dev_put", "x_prep_dev_free", "x_prep_dev_live", "x_prep_dev_take_rows"):
        try:
            entry = getattr(module, name)
        except (AttributeError, ImportError):
            if name not in ("x_prep_run", "x_prep_numeric_mode"):
                continue
            raise
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError(f"RobustScaler requires compiled {name}")
        entries[name] = entry
    entries["preprocessing_numeric_mode"] = entries["x_prep_numeric_mode"]
    entries.update(arena_binding=module, arena_binding_id=id(module))
    return entries


def connect(scaler_class, qselect):
    # glue: compiled entries and plain metadata are retained during import
    import sys
    from types import BuiltinFunctionType, MethodType
    from . import _backend
    import mmap
    from ._arena_io import DeviceRows, DeviceCache, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context, _layout
    from .tokenizer import _binding

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            connected[mode] = _entries(_backend.binding("_mojolearn_x_prep", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    base = _backend.binding("_mojolearn", "identical")
    gather, finite32, finite64 = base.gather_rows_bytes, base.all_finite_f32, base.all_finite_f64
    door = _binding()._module
    call, attribute = door.robust_scaler_native_call, door.standard_scaler_attribute_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, attribute, gather, finite32, finite64)):  # glue: fixed native ABI identity
        raise TypeError("RobustScaler requires compiled lifecycle, attribute and row-gather entries")
    context.update(arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_cache_class=DeviceCache,
                   arena_scope_dict=vars(_Scope)['__dict__'], arena_cache_dict=vars(DeviceCache)['__dict__'],
                   robust_qselect=qselect, robust_method_type=MethodType,
                   robust_fit_method=scaler_class.fit, robust_transform_method=scaler_class.transform,
                   scaler_bindings=connected, scaler_class=scaler_class, scaler_host_contexts={},
                   maxabs_rows_class=DeviceRows, maxabs_rows_owner=vars(DeviceRows)['_owner'].__get__, maxabs_gather_rows=gather,
                   maxabs_finite_f32=finite32, maxabs_finite_f64=finite64, classical_getframe=sys._getframe)
    descriptor = door.classical_attribute_descriptor(MethodType(attribute, context))
    scaler_class.__getattribute__ = descriptor
    context.update(classical_attribute_descriptor=descriptor, classical_public_layouts={},
                   classical_public_installed=True)
    context["classical_public_layouts"][scaler_class] = _layout(scaler_class, descriptor)
    # The CPU context of `device="cpu"`: the host x_prep image beside the device one.
    from ._native_prep_device import connect_cpu_device
    connect_cpu_device(context, lambda load: _entries(load("_mojolearn_x_prep_host")))
    return call, context


def native_method(method):
    # glue: only these complete methods own output/fitted/input handling;
    # application subclass overrides retain the ordinary compatibility wrappers.
    method._mojolearn_public_out = True
    method._mojolearn_tracks_fit = True
    method._mojolearn_needs_fit = True
    method._mojolearn_validates = True
    return method
