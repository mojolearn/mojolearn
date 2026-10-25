# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the x_prep imputers,
discretizer and encoders (SimpleImputer, KBinsDiscretizer, OrdinalEncoder,
OneHotEncoder, TargetEncoder, SplineTransformer, SelectKBest) and its fixed ABI."""


def connect(kinds, qselect, scores, owner_kinds=None, owner_classes=None):
    # glue: compiled entries and plain metadata are retained during import;
    # `kinds` maps each public class to the kind the owner dispatches on,
    # `qselect` is MOJOLEARN_X_PREP_FAST_QSELECT as read at import, `scores`
    # maps the built-in score functions SelectKBest runs natively to their names.
    # `owner_kinds` maps the classes the family door (bindings/prep_owner_native.mojo)
    # serves outside the encode owner to their kinds; `owner_classes` names the
    # public classes those owners construct (IterativeImputer's SimpleImputer).
    import sys
    import mmap
    from types import BuiltinFunctionType
    from . import _backend
    from ._arena_io import DeviceRows, DeviceCache, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_x_prep", mode)
            entries = {}
            for name in ("x_prep_run", "x_prep_numeric_mode", "x_prep_idn_fam", "x_prep_vendor",  # glue: fixed native exports connected during import
                         "x_prep_host_column", "x_prep_run_out", "x_prep_run_scratch", "x_prep_run_ranges",
                         "x_prep_dev_put", "x_prep_dev_free", "x_prep_dev_live", "x_prep_dev_take_rows",
                         "x_prep_cls2_cat", "x_prep_fast_unique", "x_prep_c08_routes", "x_prep_target_scratch",
                         "x_prep_spline_fused", "x_prep_label_present"):
                try:
                    entry = getattr(module, name)
                except (AttributeError, ImportError):
                    if name not in ("x_prep_run", "x_prep_numeric_mode"):
                        continue
                    raise
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"the encode prep owner requires compiled {name}")
                entries[name] = entry
            entries["preprocessing_numeric_mode"] = entries["x_prep_numeric_mode"]
            entries.update(arena_binding=module, arena_binding_id=id(module))
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    gather = _backend.binding("_mojolearn", "identical").gather_rows_bytes
    call = _binding()._module.prep_encode_native_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, gather)):  # glue: fixed native ABI identity
        raise TypeError("the encode prep owner requires compiled prep_encode_native_call and row-gather entries")
    # The owners read the process mode from _backend's namespace.
    _backend.default_mode()
    context.update(arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_cache_class=DeviceCache,
                   arena_scope_dict=vars(_Scope)['__dict__'], arena_cache_dict=vars(DeviceCache)['__dict__'],
                   scaler_bindings=connected, scaler_host_contexts={},
                   maxabs_rows_class=DeviceRows, maxabs_rows_dict=vars(DeviceRows)['__dict__'], maxabs_gather_rows=gather,
                   classical_getframe=sys._getframe, prep_encode_kinds=dict(kinds),
                   prep_encode_qselect=bool(qselect), prep_encode_scores=dict(scores),
                   prep_owner_kinds=dict(owner_kinds or {}), prep_owner_classes=dict(owner_classes or {}))
    context.update(label_context())
    # Only these exact composition classes use the native public lifecycle.
    # Keep the native instance-dictionary descriptor, not a Python getter.
    context['prep_composition_plans'] = {
        cls: dict(parameters=tuple(cls._parameters),
                  fields_get=next(vars(base)['__dict__'].__get__ for base in cls.__mro__  # glue: fixed native instance storage descriptor
                                  if '__dict__' in vars(base)))
        for cls, kind in (owner_kinds or {}).items()  # glue: fixed import-time native composition parameter/storage schemas
        if kind in ('rfe', 'iterative_imputer')
    }
    return call, context
