# SPDX-License-Identifier: Apache-2.0
"""Import-time entries and input metadata for the native model-selection owner."""

# The base binding exports the model-selection doors still look up by name:
# the base helper table's operations (lane calltable-base-helpers). Every other
# helper is called through the native model-selection helper table
# (core/msel_entry_table.mojo, key `msel_native_key` below) or linked into the
# door directly (lane calltable-msel).
_HELPERS = ('bincount2_i32', 'bincount_i64', 'check_indices_i64', 'reduce_stat')

_SPLITTER_FIELDS = ('_default_test_size', '_folds', '_pairs', 'cv', 'cvargs', 'gap', 'max_train_size',
                    'n_groups', 'n_repeats', 'n_splits', 'p', 'random_state', 'shuffle', 'test_fold', 'test_size', 'train_size')


def connect(context, resident_context):
    # glue: resolve the established identical helper ABI once; an absent
    # optional helper retains its refusal until the operation needs it.
    # math (floor/ceil/comb) is connected and imported by mojolearn._native_loaded_modules at package import.
    import operator  # noqa: F401 - glue: connect native scalar/index operators at import
    from types import BuiltinFunctionType, GetSetDescriptorType
    from ._buffer import _native
    from ._native_labels_bridge import label_context, _refusal
    entries = {}
    for name in _HELPERS:  # glue: fixed native model-selection exports
        try:
            entry = _native(name)
            if type(entry) is not BuiltinFunctionType:
                raise TypeError(f'model selection requires the compiled {name} entry')
            entries[name] = entry
        except Exception as error:
            entries[name] = _refusal(error)
    # glue: once at import, the base helper table key the native doors read
    # (lane calltable-base-helpers); a binary without one records the refusal.
    # The model-selection helper table is published by the same binary under
    # the same (mode, vendor) key, so the msel doors read this one record
    # (lane perf-audit-20261111 folded the second, identical computation).
    from ._buffer import base_native_key
    try:
        entries['base_native_key'] = base_native_key()
    except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
        entries['base_native_key'] = _refusal(error)
    entries['msel_native_key'] = entries['base_native_key']
    context.update(label_context())
    from ._native_owner_lifetime import connect_owner_context
    connect_owner_context(context)
    context.update(family_msel_entries=entries, family_msel_resident_context=resident_context)
    # The resident row owner (bindings/arena_resident_rows_door.mojo) materializes
    # a foreign fold container through this same native input owner.
    resident_context['arena_msel_input_context'] = context
    layouts = {}
    for cls in context['family_msel_splitters']:  # glue: exact supported splitter storage and lifecycle metadata
        mro = type.__dict__['__mro__'].__get__(cls)
        dictionaries = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)  # glue: retain class dictionaries once at import
        descriptor = next(fields['__dict__'] for fields in dictionaries if '__dict__' in fields)  # glue: find the native storage descriptor at import
        if type(descriptor) is not GetSetDescriptorType:
            raise TypeError('native model selection requires an ordinary splitter dictionary')
        hook_names = ('__dict__', '__getattribute__', '__getattr__', '__setattr__', '__del__')
        hooks = tuple(tuple(fields.get(name) for name in hook_names) for fields in dictionaries)  # glue: freeze lifecycle identities at import
        defaults = {}
        for fields in dictionaries:  # glue: native class constants, without invoking descriptors
            for name in _SPLITTER_FIELDS:  # glue: fixed splitter configuration names, import-time metadata only
                if name in fields:
                    defaults.setdefault(name, fields[name])
        members = tuple({name: fields.get(name) for name in _SPLITTER_FIELDS} for fields in dictionaries)  # glue: retain exact native field metadata at import
        layouts[cls] = (mro, descriptor.__get__, hooks, defaults, members)
    context['family_msel_splitter_layouts'] = layouts
