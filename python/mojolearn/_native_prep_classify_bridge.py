# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the native owner of the prep lane's classifiers
(GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB,
LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis;
bindings/prep_classify_native.mojo) and its fixed ABI."""

#: The x_prep exports the owner probes (`_optional_prep_entry`).
_OPTIONAL = ("x_prep_idn_int", "x_prep_idn_fam", "x_prep_idn_fam2", "x_prep_proba64",
             "x_prep_classical_shared", "x_prep_nb_csr_fit", "x_prep_nb_csr_jll")


def connect(kinds, prog_context):
    # glue: compiled entries and plain metadata are retained during import;
    # `kinds` maps each public class to the kind the owner dispatches on and
    # `prog_context` is the shared arena runner's (`_PROG_CONTEXT`).
    import sys
    import mmap
    import numbers
    from types import BuiltinFunctionType
    from . import _backend
    from . import _arena_io
    from ._arena_io import DeviceRows, DeviceCache, _Scope, _LOCAL
    from ._native_classical_bridge import _state_context, _numpy_free_context
    from ._native_labels_bridge import label_context
    from .tokenizer import _binding

    context = _state_context()
    context["output_context"] = _numpy_free_context()
    context.update(label_context())
    connected = {}
    for mode in ("identical", "fast"):  # glue: fixed numeric-mode ABI setup
        try:
            module = _backend.binding("_mojolearn_x_prep", mode)
            module.x_prep_run  # load and validate the family
            entries = {}
            for name in _OPTIONAL:  # glue: fixed optional native exports probed during import
                try:
                    entry = getattr(module, name)
                except (AttributeError, ImportError):
                    continue
                if not isinstance(entry, BuiltinFunctionType):
                    raise TypeError(f"the classifier owner requires compiled {name}")
                entries[name] = entry
            connected[mode] = entries
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
            connected[mode] = type(error).__name__ + ": " + str(error)
    gather = _backend.binding("_mojolearn", "identical").gather_rows_bytes
    call = _binding()._module.prep_classify_native_call
    if any(not isinstance(entry, BuiltinFunctionType) for entry in (call, gather)):  # glue: fixed native ABI identity
        raise TypeError("the classifier owner requires compiled prep_classify_native_call and row-gather entries")
    # The owner reads the process mode from _backend's namespace.
    _backend.default_mode()
    context.update(arena_mmap=mmap.mmap, arena_local=_LOCAL,
                   arena_scope_class=_Scope, arena_cache_class=DeviceCache,
                   arena_scope_dict=vars(_Scope)['__dict__'], arena_cache_dict=vars(DeviceCache)['__dict__'],
                   maxabs_rows_class=DeviceRows, maxabs_rows_dict=vars(DeviceRows)['__dict__'], maxabs_gather_rows=gather,
                   classical_getframe=sys._getframe, cls_kinds=dict(kinds), cls_bindings=connected,
                   prog_context=prog_context, real_type=numbers.Real, integral_type=numbers.Integral,
                   resident_class=_arena_io.resident, resident_min_words=_arena_io.RESIDENT_MIN_WORDS,
                   arena_io_fields=vars(_arena_io))
    return call, context
