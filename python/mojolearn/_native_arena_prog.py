# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native arena program runner.

`_expansion_prep._Prog.run` and `_expansion_metrics._Prog.run` are one call
into `arena_prog_native_call` (bindings/arena_prog_native.mojo), which owns
the run: the binding entries, the arena and output stores, the input copies,
resident and device-gathered inputs, range descriptors and stage words.
"""

#: family -> (binding module, entry prefix, mandatory entry, optional entries,
#: the refusal of a numeric_mode that is not 'fast' or 'identical')
_FAMILIES = {
    "x_prep": ("_mojolearn_x_prep", "x_prep_run",
               ("x_prep_run_out", "x_prep_run_scratch", "x_prep_run_ranges",
                "x_prep_dev_put", "x_prep_dev_free", "x_prep_dev_live", "x_prep_dev_take_rows"),
               "numeric_mode must be fast, identical or None"),
    "x_metrics": ("_mojolearn_x_metrics", "x_metrics_run",
                  ("x_metrics_run_out", "x_metrics_run_ranges"),
                  "numeric_mode must be 'fast' or 'identical'"),
}

#: the Python class a stored connection error is raised as (first match)
_ERROR_CLASSES = ("ImportError", "AttributeError", "NotImplementedError", "ValueError", "TypeError", "RuntimeError")


def _error_text(error):
    # glue: name the builtin category of a connection error once, at import
    for cls in type(error).__mro__:  # glue: the error class's fixed ancestry
        if cls.__name__ in _ERROR_CLASSES and cls.__module__ == "builtins":
            return cls.__name__ + ": " + str(error)
    return "RuntimeError: " + str(error)


def connect(family, prog_class, scratch_class=None, op_names=None):
    # glue: compiled entries and plain metadata are retained during import
    import mmap
    from types import BuiltinFunctionType
    from . import _backend
    from ._arena_io import DeviceRows, DeviceCache, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from .tokenizer import _binding

    module_name, mandatory, optional, refusal = _FAMILIES[family]
    _backend.default_mode()  # glue: resolve the package default mode once, so the native run reads it
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding(module_name, mode)
            entries = {mandatory: getattr(module, mandatory)}
            for name in optional:  # glue: fixed optional native exports connected during import
                try:
                    entries[name] = getattr(module, name)
                except (AttributeError, ImportError):
                    continue
            for name, entry in entries.items():  # glue: admit only compiled entries
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"arena programs require compiled {name}")
            entries.update(prog_family=family, arena_binding=module, arena_binding_id=id(module))
            connected[mode] = entries
        except Exception as error:  # glue: the per-mode connection error is raised by the run that asks for that mode
            connected[mode] = _error_text(error)
    base = _backend.binding("_mojolearn", "identical")
    gather, finite32, finite64 = base.gather_rows_bytes, base.all_finite_f32, base.all_finite_f64
    call = _binding()._module.arena_prog_native_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, gather, finite32, finite64)):  # glue: fixed native ABI identity
        raise TypeError("arena programs require the compiled program runner and row-gather entries")
    context = _state_context()
    context["output_context"] = _numpy_free_context()
    context.update(prog_class=prog_class, prog_bindings=connected, prog_mode_refusal=refusal,
                   prog_mmap=mmap.mmap, prog_op_names=dict(op_names or {}),
                   prog_types=dict(int=int, str=str, bytes=bytes, memoryview=memoryview),
                   arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_cache_class=DeviceCache,
                   arena_scope_dict=vars(_Scope)["__dict__"], arena_cache_dict=vars(DeviceCache)["__dict__"],
                   maxabs_rows_class=DeviceRows, maxabs_rows_dict=vars(DeviceRows)["__dict__"],
                   maxabs_gather_rows=gather, maxabs_finite_f32=finite32, maxabs_finite_f64=finite64)
    if scratch_class is not None:
        context.update(prog_scratch_class=scratch_class,
                       prog_scratch_off=vars(scratch_class)["off"].__get__,
                       prog_scratch_kind=vars(scratch_class)["kind"].__get__)
    return call, context
