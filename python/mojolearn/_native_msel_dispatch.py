# SPDX-License-Identifier: Apache-2.0
"""Import-time estimator ABI metadata for compiled cross-validation."""
from types import BuiltinFunctionType, GetSetDescriptorType


def connect(context, package):
    # Called after package classes are defined; nothing here runs in CV.
    # Re-run when a family loads (lane lazy-family-import, 2026-11-12): each
    # block registers the classes of a LOADED family once; a class already in
    # the registry keeps the plan taken when its family first connected.
    import sys
    from . import _backend
    from . import _arena_io
    from _json import encode_basestring_ascii

    def loaded(name):
        return sys.modules.get(package.__name__ + '.' + name)

    linear_model, decomposition = loaded('linear_model'), loaded('decomposition')
    preprocessing, _expansion_prep = loaded('preprocessing'), loaded('_expansion_prep')
    resample = loaded('resample')
    _backend.default_mode()  # glue: settle the shared default before native metadata reads
    if resample is not None and 'family_msel_fast_masks' not in context:
        context["family_msel_fast_masks"] = {mode: resample.fast_defines(mode) for mode in ("identical", "fast")}  # glue: existing compiled option masks, no product resolver callback
    context.update(family_msel_rows_entry=_arena_io._ROWS_CALL,
                   family_msel_rows_context=_arena_io._ROWS_CONTEXT,
                   family_msel_json_string=encode_basestring_ascii)
    registry = context.setdefault('family_msel_estimators', {})
    protected = ('__new__', '__init__', '__getattribute__', '__getattr__', '__setattr__', '__del__',
                 '__hash__', '__eq__', '__ne__',
                 '__sklearn_clone__', 'get_params', 'set_params', 'fit', 'predict', 'predict_proba',
                 'decision_function', 'transform', 'score', 'split', 'get_n_splits', '_parameter_names',
                 '_has_constructor_parameters', '_BINDING', '_mojolearn_device_rows', '_estimator_type')
    class_fields = ('classes_', 'coef_', 'feature_importances_', '_power', '_loss_name', '_nodes', '_hyper', '_replace')

    def register(cls, entry, owner_context, parameters, init, methods, forest=None):
        # glue: fixed class layout, parameter names and native method ABIs.
        # True when newly registered; an already connected class is kept.
        if cls in registry:
            return False
        mro = type.__dict__['__mro__'].__get__(cls)
        fields = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)  # glue: import-time class metadata
        descriptor = next((row['__dict__'] for row in fields if '__dict__' in row), None)  # glue: native storage layout
        if type(cls) is not type or type(descriptor) is not GetSetDescriptorType:
            raise TypeError('native CV requires an ordinary native instance dictionary')
        hooks = tuple((base, name, row.get(name)) for base, row in zip(mro, fields)  # glue: retain method/layout identities once
                      for name in protected + class_fields)  # glue: fixed import-time method identity inventory
        registry[cls] = dict(entry=entry if type(entry) is BuiltinFunctionType else None,
                             context=owner_context, parameters=tuple(parameters), init=init,
                             methods=methods, forest=forest, mro=mro, hooks=hooks,
                             fields_get=descriptor.__get__, protected=protected, classifier=getattr(cls, '_estimator_type', None) == 'classifier',
                             estimator_type=getattr(cls, '_estimator_type', None),
                             device_rows=bool(getattr(cls, '_mojolearn_device_rows', False)))
        return True

    # Each recipe contains argument positions or fixed values. Native code
    # builds the ABI tuple; it never calls a Python method/constructor.
    arg = lambda at: ('arg', at)
    const = lambda value: ('constant', value)
    for cls, names, fit in (() if linear_model is None else (  # glue: two native linear schemas
            (linear_model.LinearRegression, ('fit_intercept', 'numeric_mode'), 'ols_fit'),
            (linear_model.Ridge, ('alpha', 'solver', 'fit_intercept', 'normalize', 'numeric_mode'), 'ridge_fit'))):
        register(cls, linear_model._native_classical, linear_model._CLASSICAL, names, 'initialize', {
            'fit': (fit, (arg(0), arg(1), const(None))),
            'predict': ('predict_public', (arg(0),)),
            'score': ('score_public', (arg(0), arg(1))),
            'get_params': ('get_params', (const(False),)),
            'set_params': ('set_params', (arg(0),)),
        })
        # The old Python scorer was an internal caller and consumed the
        # float32 kernel output. A native scorer has no Python package frame;
        # select that same output explicitly without changing public predict.
        registry[cls]['scorer_predict_operation'] = 'predict_score'
        if cls is linear_model.Ridge:
            registry[cls]['methods']['validate_params'] = ('validate_params', ())
    if decomposition is not None:
        register(decomposition.PCA, decomposition._PCA_ENTRY, decomposition._PCA_CONTEXT,
                 ('n_components', 'copy', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'n_oversamples',
                  'power_iteration_normalizer', 'random_state', 'numeric_mode'), 'initialize', {
                     'fit': ('fit', (arg(0), arg(1))), 'transform': ('transform', (arg(0),)),
                     'get_params': ('get_params', (const(False),)),
                     'set_params': ('set_params', (arg(0),)),
                 })
    scalers = ()
    if preprocessing is not None:
        scalers += ((preprocessing.StandardScaler, 'STANDARD', ('copy', 'with_mean', 'with_std', 'numeric_mode'), (arg(0), const(False), const(None))),
                    (preprocessing.MinMaxScaler, 'MINMAX', ('feature_range', 'copy', 'clip', 'numeric_mode'), (arg(0), const(False))))
    if _expansion_prep is not None:
        scalers += ((_expansion_prep.MaxAbsScaler, 'MAXABS', ('copy',), (arg(0),)),
                    (_expansion_prep.RobustScaler, 'ROBUST', ('with_centering', 'with_scaling', 'quantile_range', 'copy', 'unit_variance'), (arg(0),)))
    for cls, owner, names, transform in scalers:  # glue: fixed native scaler schemas
        module = preprocessing if owner in ('STANDARD', 'MINMAX') else _expansion_prep
        register(cls, vars(module)['_' + owner + '_CALL'], vars(module)['_' + owner + '_CONTEXT'], names, 'init', {
            'fit': ('fit', (arg(0), arg(1))), 'transform': ('transform', transform),
            'get_params': ('get_params', (const(False),)),
            'set_params': ('set_params', (arg(0),)),
        })
    if preprocessing is not None:
        registry[preprocessing.StandardScaler]['fit_entry'] = preprocessing._STANDARD_FIT
    forests = tuple(getattr(module, name) for module, names in (  # glue: exact native forest schemas of loaded families
        (loaded('randomforest'), ('RandomForestClassifier', 'RandomForestRegressor')),
        (loaded('extratrees'), ('ExtraTreesClassifier', 'ExtraTreesRegressor')),
        (loaded('tree'), ('DecisionTreeClassifier', 'DecisionTreeRegressor')))
        if module is not None for name in names)
    if any(cls not in registry for cls in forests):
        from . import _native_trees_bridge
        entry, trees = _native_trees_bridge.connect()
        for cls in forests:  # glue: exact native forest schemas
            if register(cls, entry, trees, getattr(cls, '_parameter_names', ()), None, {
                    'fit': ('trees_forest_fit', (arg(0), arg(1), const(None), const(False), const(None))),
                    'predict': ('trees_forest_predict', (arg(0),)),
                    'predict_proba': ('trees_forest_predict_proba', (arg(0),)),
                    'score': ('trees_score_forest', (arg(0), arg(1), const(None))),
                    }, trees.get('trees_classes', {}).get(cls)) and not registry[cls]['classifier']:
                registry[cls]['methods'].pop('predict_proba')
    # Preserve explicit CPU estimator contexts if the package connected them.
    host = linear_model._CLASSICAL.get('classical_host_context') if linear_model is not None else None
    if host is not None:
        for name, cls in host.get('classical_host_estimators', {}).items():  # glue: already-connected explicit CPU classes
            base = registry[getattr(linear_model, name)]
            register(cls, linear_model._native_classical, host, base['parameters'], base['init'], base['methods'])
            registry[cls]['scorer_predict_operation'] = base['scorer_predict_operation']
    from ._native_msel_extra import connect as connect_extra
    connect_extra(context, register, registry)
    from ._native_msel_prep import connect as connect_prep
    connect_prep(context, register, registry)
    neighbors = loaded('neighbors')
    if neighbors is not None:
        registry[neighbors.KNeighborsClassifier]["classes_kind"] = "knn"  # glue: native form of the connected class-label property
    # glue: the DeviceRows door (bindings/device_rows_door.mojo) reads these
    # connected identities from the owner context a door receives; a
    # context without them treats no argument as DeviceRows. door_rows_call
    # (lane a6) resolves a fold's resident store ids for a device kernel
    # entry (bindings/device_rows_door.mojo door_rows_store).
    door = dict(door_rows_class=_arena_io.DeviceRows, door_rows_context=_arena_io._ROWS_CONTEXT,
                door_rows_call=_arena_io._ROWS_CALL)
    for plan in registry.values():  # glue: import-time connection of fixed identities into each owner context
        if type(plan['context']) is dict:
            plan['context'].update(door)
    context.update(family_msel_estimators=registry,
                   family_msel_pending_estimators=tuple(value for value in vars(package).values()  # glue: distinguish known migration debt from external callbacks
                                                        if type(value) is type and value not in registry),
                   family_msel_backend_fields=vars(_backend))

    selection = loaded('model_selection')
    if 'family_msel_metric_plans' not in context:
        from ._native_msel_scorers import connect as connect_scorers
        connect_scorers(context, selection)

    if 'family_msel_search_layouts' not in context:
        from ._native_msel_search import connect as connect_search
        connect_search(context, selection)

    from ._native_parallel_cv import connect as connect_parallel
    connect_parallel(context, package)
