# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the stateless x_prep
transformers (Binarizer, Normalizer, PolynomialFeatures, VarianceThreshold and
the selectors' transform) and its fixed ABI."""

CONTEXT_CONNECTION = None


def connect(kinds):
    global CONTEXT_CONNECTION
    # glue: compiled entries and plain metadata are retained during import;
    # `kinds` maps each public class to the kind the owner dispatches on.
    import sys
    import mmap
    from types import BuiltinFunctionType
    from . import _backend
    from ._arena_io import DeviceRows, DeviceCache, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from .tokenizer import _binding
    from ._native_loaded_modules import connect_module

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_x_prep", mode)
            entries = {}
            for name in ("x_prep_run", "x_prep_numeric_mode", "x_prep_idn_fam",  # glue: fixed native exports connected during import
                         "x_prep_run_out", "x_prep_run_scratch", "x_prep_run_ranges",
                         "x_prep_dev_put", "x_prep_dev_free", "x_prep_dev_live", "x_prep_dev_take_rows",
                         "x_prep_host_column", "x_prep_ptimpute_flags"):
                try:
                    entry = getattr(module, name)
                except (AttributeError, ImportError):
                    if name not in ("x_prep_run", "x_prep_numeric_mode"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"the stateless prep owner requires compiled {name}")
                entries[name] = entry
            entries["preprocessing_numeric_mode"] = entries["x_prep_numeric_mode"]
            entries.update(arena_binding=module, arena_binding_id=id(module))
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    gather = _backend.binding("_mojolearn", "identical").gather_rows_bytes
    call = _binding()._module.prep_stateless_native_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, gather)):  # glue: fixed native ABI identity
        raise TypeError("the stateless prep owner requires compiled prep_stateless_native_call and row-gather entries")
    # The owners read the process mode from _backend's namespace.
    _backend.default_mode()
    context.update(arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_cache_class=DeviceCache,
                   arena_scope_dict=vars(_Scope)['__dict__'], arena_cache_dict=vars(DeviceCache)['__dict__'],
                   scaler_bindings=connected, scaler_host_contexts={},
                   maxabs_rows_class=DeviceRows, maxabs_rows_dict=vars(DeviceRows)['__dict__'], maxabs_gather_rows=gather,
                   classical_getframe=sys._getframe, prep_stateless_kinds=dict(kinds))
    # glue: immutable identities of the compiled exports actually selected
    # above; no assumed function name/module-owner convention is required.
    context["prep_stateless_entry_identities"] = tuple(
        (mode, tuple(table.items()) if type(table) is dict else None)
        for mode, table in connected.items()  # glue: freeze fixed import-time native ABI identities
    )
    # glue: freeze native class/descriptor references and fixed mapping contents
    # during package connection. Native operations copy these records and never
    # execute this setup helper. Process namespaces keep their pinned identity.
    static_names = ("array_fields", "array_storage_slots", "buffer_codes",
                    "prep_stateless_kinds", "warning_category_slots")
    output = context["output_context"]
    CONTEXT_CONNECTION = (
        context, tuple(context.items()),
        tuple((key, context[key], tuple(context[key].items()))
              for key in static_names if key in context),  # glue: fixed import-time provider maps
        output, tuple(output.items()),
        tuple((key, output[key], tuple(output[key].items()))
              for key in static_names if key in output),  # glue: fixed import-time output provider maps
    )
    connect_module(__name__, sys.modules[__name__], ("CONTEXT_CONNECTION",))
    return call, context
