# SPDX-License-Identifier: Apache-2.0
"""Import-time entries and input metadata for the native model-selection owner."""

_HELPERS = (
    'arange_i64', 'bincount2_i32', 'bincount_i64', 'cast_elements', 'check_indices_i64',
    'count_mask_u8', 'first_seen_i32', 'fold_ids', 'gather_i32', 'gather_i64',
    'group_fold_assign_i32', 'indices_overlap_i64', 'leave_range_i64', 'mask_from_indices_u8',
    'msel_alloc', 'msel_free', 'msel_group_fold_perm_i32', 'msel_proba_column', 'msel_read', 'msel_scatter_rows',
    'msel_split_table_i32', 'reduce_stat', 'row_means_f64', 'row_stds_f64', 'select_fold_i64',
    'select_mask_u8_i64', 'strat_alloc_i64', 'strat_fold_assign_i32', 'strat_group_plan_i32',
)

_SPLITTER_FIELDS = ('_default_test_size', '_folds', '_pairs', 'cv', 'cvargs', 'gap', 'max_train_size',
                    'n_groups', 'n_repeats', 'n_splits', 'p', 'random_state', 'shuffle', 'test_size', 'train_size')


def connect(context, resident_context):
    # glue: resolve the established identical helper ABI once; an absent
    # optional helper retains its refusal until the operation needs it.
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
    context.update(label_context())
    context.update(family_msel_entries=entries, family_msel_resident_context=resident_context)
    layouts = {}
    for cls in context['family_msel_splitters']:  # glue: exact supported splitter storage and lifecycle metadata
        mro = type.__dict__['__mro__'].__get__(cls)
        dictionaries = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)
        descriptor = next(fields['__dict__'] for fields in dictionaries if '__dict__' in fields)
        if type(descriptor) is not GetSetDescriptorType:
            raise TypeError('native model selection requires an ordinary splitter dictionary')
        hooks = tuple(tuple(fields.get(name) for name in ('__dict__', '__getattribute__', '__getattr__',
                                                         '__setattr__', '__del__')) for fields in dictionaries)
        defaults = {}
        for fields in dictionaries:  # glue: native class constants, without invoking descriptors
            for name in _SPLITTER_FIELDS:
                if name in fields:
                    defaults.setdefault(name, fields[name])
        members = tuple({name: fields.get(name) for name in _SPLITTER_FIELDS} for fields in dictionaries)
        layouts[cls] = (mro, descriptor.__get__, hooks, defaults, members)
    context['family_msel_splitter_layouts'] = layouts
