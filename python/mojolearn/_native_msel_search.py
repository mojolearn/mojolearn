# SPDX-License-Identifier: Apache-2.0
"""Import-time native search/container storage layouts."""
from types import FunctionType, GetSetDescriptorType


def connect(context, selection):
    # glue: fixed class storage and hook metadata, connected once at import
    schemas = (
        (selection._BaseSearch, ('estimator', 'scoring', 'n_jobs', 'refit', 'cv', 'verbose',
                                 'pre_dispatch', 'error_score', 'return_train_score')),
        (selection.GridSearchCV, ('estimator', 'param_grid', 'scoring', 'n_jobs', 'refit', 'cv',
                                  'verbose', 'pre_dispatch', 'error_score', 'return_train_score')),
        (selection.RandomizedSearchCV, ('estimator', 'param_distributions', 'n_iter', 'scoring',
                                        'n_jobs', 'refit', 'cv', 'verbose', 'pre_dispatch',
                                        'random_state', 'error_score', 'return_train_score')),
        (selection.ParameterGrid, ('param_grid',)),
        (selection.ParameterSampler, ('param_distributions', 'n_iter', 'random_state')),
    )
    layouts = {}
    for cls, names in schemas:  # glue: connect fixed public native storage types
        mro = type.__dict__['__mro__'].__get__(cls)
        rows = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)  # glue: retain native class dictionaries
        descriptor = next((row['__dict__'] for row in rows if '__dict__' in row), None)  # glue: ordinary instance storage
        if type(cls) is not type or type(descriptor) is not GetSetDescriptorType:
            raise TypeError('native model selection requires ordinary instance storage')
        protected = tuple(sorted(set(('__getattribute__', '__getattr__', '__setattr__', '__del__',  # glue: import-time class-layout inventory
                                      '__new__', '__init__', '__dict__')) |
                                 {name for row in rows for name, value in row.items()  # glue: fixed method/property identity inventory
                                  if type(value) in (FunctionType, property)}))
        hooks = tuple((base, name, row.get(name)) for base, row in zip(mro, rows)  # glue: fixed ancestry and hook snapshots
                      for name in protected)  # glue: refuse changed class hooks before reading state
        layouts[cls] = dict(parameters=names, mro=mro, hooks=hooks,
                            protected=protected, fields_get=descriptor.__get__)
    context['family_msel_search_layouts'] = layouts
