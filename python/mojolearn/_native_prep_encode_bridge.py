# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the x_prep imputers,
discretizer and encoders (SimpleImputer, KBinsDiscretizer, OrdinalEncoder,
OneHotEncoder, TargetEncoder, SplineTransformer, SelectKBest) and its fixed ABI."""


def connect(kinds, qselect, scores, owner_kinds=None, owner_classes=None):
    # glue: compiled entries and plain metadata are retained during import;
    # `kinds` maps each public class to the kind the owner dispatches on,
    # `qselect` is MOJOLEARN_X_PREP_FAST_QSELECT as read at import, `scores`
    # maps the built-in score functions SelectKBest runs natively to their names.
    # `owner_kinds` maps the classes the family door (bindings/prep_owner_native_door.mojo)
    # serves outside the encode owner to their kinds; `owner_classes` names the
    # public classes those owners construct (IterativeImputer's SimpleImputer).
    import sys
    import mmap
    from types import BuiltinFunctionType
    from . import _backend
    from ._arena_io import DeviceRows, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    # Native warning delivery (bindings/classical_warnings_unpack.mojo): filters read
    # natively, standard display hooks only, no warnings.warn call.
    from ._native_classical_warnings import connect_warnings
    context.update(connect_warnings())
    context["output_context"] = _numpy_free_context()
    def image_entries(module):
        # glue: the fixed x_prep exports of one compiled image, connected during import
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
        return entries

    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            connected[mode] = image_entries(_backend.binding("_mojolearn_x_prep", mode))
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    base = _backend.binding("_mojolearn", "identical")
    gather, finite32, finite64 = base.gather_rows_bytes, base.all_finite_f32, base.all_finite_f64
    call = _binding()._module.prep_encode_native_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, gather, finite32, finite64)):  # glue: fixed native ABI identity
        raise TypeError("the encode prep owner requires compiled prep_encode_native_call, row-gather and finite-check entries")
    # The owners read the process mode from _backend's namespace.
    _backend.default_mode()
    context.update(arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_scope_owner=vars(_Scope)['_owner'].__get__,
                   scaler_bindings=connected, scaler_host_contexts={},
                   maxabs_rows_class=DeviceRows, maxabs_rows_owner=vars(DeviceRows)['_owner'].__get__, maxabs_gather_rows=gather,
                   # Shared composition target validation preserves both widths.
                   maxabs_finite_f32=finite32, maxabs_finite_f64=finite64,
                   classical_getframe=sys._getframe, prep_encode_kinds=dict(kinds),
                   prep_encode_qselect=bool(qselect), prep_encode_scores=dict(scores),
                   prep_owner_kinds=dict(owner_kinds or {}), prep_owner_classes=dict(owner_classes or {}))
    context.update(label_context())
    # `device=` of the classes this door serves (bindings/prep_owner_route_unpack.mojo):
    # the builtins module the shared compiled device check takes, and, beside
    # a device set, the host image's entry table as the identical binding of
    # the CPU route. The CPU context itself is built by the compiled resolver
    # at the first CPU route of the process, because this context is completed
    # after this function returns. A host image that is not built is kept as
    # its refusal text, said when a `device="cpu"` estimator first runs.
    import builtins
    from ._native_prep_device import _FAST_ON_CPU
    context["prep_builtins"] = builtins
    if context["vendor"] != "cpu":
        try:
            context["prep_cpu_bindings"] = {
                "identical": image_entries(_backend.load_host_module("_mojolearn_x_prep_host")), "fast": _FAST_ON_CPU}
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            context["prep_cpu_bindings"] = type(error).__name__ + ": " + str(error)
    # mutual_info's discrete_features indices (bindings/prep_select_native_door.mojo
    # `_discrete_mask`): an exact int matches `int` by CPython's type test and
    # never enters numbers.Integral's Python ABCMeta.__instancecheck__ (census
    # d95f805cce: 2 frames per index).
    from ._native_loaded_modules import INTEGRAL_TYPES
    context["integral_type"] = INTEGRAL_TYPES
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
