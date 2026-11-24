# SPDX-License-Identifier: Apache-2.0
"""Exact native schemas for established boosting and naive-Bayes CV callers."""


def connect(context, register, registry):
    # glue: fixed built-in schemas and native owner metadata, once per loaded
    # family (lane lazy-family-import)
    import sys
    from ._native_classical_bridge import _layout
    boosting = sys.modules.get(__package__ + '._gbdt_adapters')
    prep = sys.modules.get(__package__ + '._expansion_prep')

    def owner_for(cls, source, names, consumed, reset):
        # glue: private context copy leaves other families' owner metadata intact
        # The trees connection context's copy follows its later CPU table
        # load (`_native_trees_bridge.derive`): a copy taken before the first
        # device="cpu" admission held no CPU table set, and a device="cpu"
        # GradientBoostingRegressor in cross_val_score was refused ("device='cpu'
        # is not connected", proving suite 2026-11-22).
        from . import _native_trees_bridge
        owner = _native_trees_bridge.derive(source) if source is _native_trees_bridge.connect()[1] else dict(source)
        layout = _layout(cls, None)
        owner['cv_fixed_owner'] = dict(
            **{'class': cls}, name=cls.__name__, parameters=tuple(names), consumed=tuple(consumed),
            reset=reset, sequence_parameters=('priors', 'class_weights'), fields_get=layout['fields_get'],
            admission={'classical_public_layouts': {cls: layout}, 'classical_public_installed': True})
        return owner

    if boosting is not None and boosting.GradientBoostingRegressor not in registry:
        _connect_boosting(boosting, register, owner_for, _layout)
    if prep is not None and prep.GaussianNB not in registry:
        _connect_naive_bayes(prep, register, owner_for)


def _connect_boosting(boosting, register, owner_for, _layout):
    # glue: both boosting adapters' fixed schemas, once, when the family loads
    cls = boosting.GradientBoostingRegressor
    names = boosting._TREES_CONTEXT['trees_gbdt_adapter_tables']['names']
    owner = owner_for(cls, boosting._TREES_CONTEXT, names,
                      ('numeric_mode_', 'n_features_in_', 'model_', 'loss_curve_', 'test_loss_curve_',
                       'best_iteration_', 'stopped_early_', 'learning_rate_'), True)
    learner = boosting.GradientBoosting
    learner_layout = _layout(learner, None)
    owner['cv_fixed_owner'].update(loss=vars(cls)['_loss'], learner_class=learner,
        learner_admission={'classical_public_layouts': {learner: learner_layout}, 'classical_public_installed': True},
        learner_fields_get=learner_layout['fields_get'],
        learner_consumed=('loss', 'numeric_mode', 'model_', 'n_features_in_', 'approx_dim_', 'n_classes_'))
    register(cls, boosting._TREES, owner, names, None, {
        'fit': ('trees_gbdt_adapter_regressor_fit', (('arg', 0), ('arg', 1), ('constant', None), ('constant', None))),
        'predict': ('trees_gbdt_adapter_decision', (('arg', 0),)),
        'score': ('trees_gbdt_adapter_score', (('arg', 0), ('arg', 1), ('constant', None))),
        'set_params': ('trees_gbdt_adapter_set_params', (('arg', 0),)),
    })

    # The established installed parallel-CV verifier covers both binary
    # boosting classification and regression through the same native owner.
    cls = boosting.GradientBoostingClassifier
    owner = owner_for(cls, boosting._TREES_CONTEXT, names,
                      ('numeric_mode_', 'n_features_in_', 'model_', 'loss_curve_', 'test_loss_curve_',
                       'best_iteration_', 'stopped_early_', 'learning_rate_', 'classes_', 'n_classes_',
                       '_label_kind_'), True)
    owner['cv_fixed_owner'].update(loss=vars(cls)['_loss'], learner_class=learner,
        learner_admission={'classical_public_layouts': {learner: learner_layout}, 'classical_public_installed': True},
        learner_fields_get=learner_layout['fields_get'],
        learner_consumed=('loss', 'numeric_mode', 'model_', 'n_features_in_', 'approx_dim_', 'n_classes_'))
    register(cls, boosting._TREES, owner, names, None, {
        'fit': ('trees_gbdt_adapter_classifier_fit', (('arg', 0), ('arg', 1), ('constant', None), ('constant', None))),
        'predict': ('trees_gbdt_adapter_predict', (('arg', 0),)),
        'predict_proba': ('trees_gbdt_adapter_proba', (('arg', 0),)),
        'decision_function': ('trees_gbdt_adapter_decision', (('arg', 0),)),
        'score': ('trees_gbdt_adapter_score', (('arg', 0), ('arg', 1), ('constant', None))),
        'set_params': ('trees_gbdt_adapter_set_params', (('arg', 0),)),
    })


def _connect_naive_bayes(prep, register, owner_for):
    # glue: GaussianNB's fixed schema, once, when the prep family loads
    cls = prep.GaussianNB
    names = ('priors', 'var_smoothing', 'numeric_mode')
    owner = owner_for(cls, prep._CLS_CONTEXT, names,
                      ('classes_', 'n_features_in_', 'numeric_mode_', 'theta_', 'var_', 'class_count_',
                       'class_prior_', 'epsilon_', '_const', '_raw_var'), False)
    register(cls, prep._CLS_CALL, owner, names, None, {
        'fit': ('fit', (('arg', 0), ('arg', 1), ('constant', None))),
        'predict': ('predict', (('arg', 0),)),
        'predict_proba': ('predict_proba', (('arg', 0),)),
        'predict_log_proba': ('predict_log_proba', (('arg', 0),)),
        'score': ('score', (('arg', 0), ('arg', 1))),
        'set_params': ('cv_set_params', (('arg', 0),)),
    })


def _mlp_methods(cls):
    # glue: the MLP owner's fixed operation names (python/mojolearn/_x_sequence_mlp.py)
    methods = {'fit': ('mlp_fit', (('arg', 0), ('arg', 1))), 'predict': ('mlp_predict', (('arg', 0),))}
    if vars(cls).get('_estimator_type') == 'classifier':
        methods['predict_proba'] = ('mlp_predict_proba', (('arg', 0),))
        methods['score'] = ('mlp_classifier_score', (('arg', 0), ('arg', 1), ('constant', None)))
    else:
        methods['score'] = ('mlp_score', (('arg', 0), ('arg', 1), ('constant', None)))
    return methods


def _pls_methods(cls):
    # glue: the PLS owner's fixed operation names (python/mojolearn/_expansion_decomp.py)
    return {'fit': ('fit', (('arg', 0), ('arg', 1), ('constant', None))),
            'predict': ('predict', (('arg', 0),)),
            'transform': ('transform', (('arg', 0), ('constant', None), ('constant', None))),
            'score': ('score', (('arg', 0), ('arg', 1), ('constant', None)))}


def _rnn_methods(cls):
    # glue: the recurrent owner's fixed operation names (python/mojolearn/_x_sequence_rnn.py)
    methods = {'fit': ('rnn_fit', (('arg', 0), ('arg', 1))), 'predict': ('rnn_predict', (('arg', 0),))}
    if getattr(cls, '_estimator_type', None) == 'classifier':
        methods['predict_proba'] = ('rnn_predict_proba', (('arg', 0),))
        methods['score'] = ('rnn_classifier_score', (('arg', 0), ('arg', 1), ('constant', None)))
    else:
        methods['score'] = ('rnn_score', (('arg', 0), ('arg', 1), ('constant', None)))
    return methods


# Member family number (core/member_families.mojo) -> the names of the family owner's entry and context in the
# module that defines the class, its method recipes and (optional fourth word) the owner's constructor operation.
# A row is NEVER required for a marked class to be registered: it only adds the Python-unit method recipes (the
# path the class takes when its member cannot serve a request) for a family whose classes had no schema of their
# own. The CLASSES come from their marks (`_mojolearn_member`), never from a list here. Without a fourth word the
# constructor only stores its parameters under their own names (init None: the native clone copies them).
_MEMBER_SCHEMAS = {
    19: ('_XSEQ', '_XSEQ_CONTEXT', _mlp_methods),
    20: ('_XSEQ', '_XSEQ_CONTEXT', _rnn_methods),
    22: ('_PLS_CALL', '_PLS_CONTEXT', _pls_methods),
}


def _derive_member(cls, fields, names):
    # glue: import-time class metadata of one marked class without a schema. Answers (entry, owner context) read
    # from the class's OWN fit door, or None when that cannot be read exactly:
    # - the owner context is the one `*_CONTEXT` dictionary of the class's module that its fit names (the context
    #   the family's member admit receives is therefore the one its fit door receives);
    # - the constructor must only store its parameters under their own names (every name its code uses is a
    #   parameter), because a member-only plan clones by copying those fields. A NATIVE constructor
    #   (python/mojolearn/_native_full_reset_params.py and _native_declared_constructors.py `_initializer`: the
    #   class's __init__ replaced at import by a direct native call) is read from its own write schema instead: it
    #   qualifies when every write is a parameter stored under its own name or a constant (the constants become the
    #   plan's initial fields, as the hand-written schemas of python/mojolearn/_native_msel_prep.py give them).
    #   Proving suite 2026-11-22: every marked class of those constructors (the naive Bayes kinds, the kernel
    #   approximations, the encoders, ...) was left unregistered, so no meta-estimator reached its member. The
    #   answer's third item is those initial fields.
    from types import BuiltinFunctionType, FunctionType
    rows = tuple(vars(base) for base in cls.__mro__[:-1])  # glue: the class's own ancestry, import-time
    fit = next((row['fit'] for row in rows if 'fit' in row), None)
    init = next((row['__init__'] for row in rows if '__init__' in row), None)
    fit = getattr(fit, '__wrapped__', fit)
    if type(fit) is not FunctionType or type(init) is not FunctionType or fit.__globals__ is not fields:
        return None
    initial = {}
    if init.__code__.co_freevars == ('owner',) and init.__closure__[0].cell_contents is cls:
        schema = init.__globals__.get('_CONTEXT', {}).get('plans', {}).get(cls)  # glue: the native constructor's schema
        if schema is None:
            return None
        stored = set()
        for field, kind, value, _rule in schema['events']:  # glue: one class's fixed constructor writes, import-time
            if kind == 'parameter' and field == value:
                stored.add(field)
            elif kind == 'constant':
                if field not in names:
                    initial[field] = value
            else:
                return None
        if not set(names) <= stored:
            return None
    elif not set(init.__code__.co_names) <= set(names) or init.__code__.co_freevars:
        return None
    used = fit.__code__.co_names
    contexts = [name for name in used if name.endswith('_CONTEXT') and type(fields.get(name)) is dict]  # glue: one fit door
    entries = [name for name in used if type(fields.get(name)) is BuiltinFunctionType]  # glue: one fit door
    if len(contexts) != 1:
        return None
    return (fields[entries[0]] if len(entries) == 1 else None), fields[contexts[0]], initial


def connect_members(context, register, registry):
    # glue: every class of a LOADED package module that carries its OWN member mark and has no schema yet is
    # registered, once per module (a family that loads later is scanned when it connects). Called AFTER every
    # hand-written schema is connected (python/mojolearn/_native_msel_dispatch.py), so a class with a schema of its
    # own keeps it and is never listed as waiting.
    # 1. A family with a row in _MEMBER_SCHEMAS: the row's owner and method recipes (member first, Python units
    #    when the member cannot serve the request).
    # 2. Any other family: a MEMBER-ONLY plan derived from the class (`_derive_member`): no method recipes, so the
    #    class runs through its member wherever the family's CAPABILITIES serve the request and is refused in the
    #    words it had before its mark everywhere else (plan['member_only']; bindings/msel_estimator_native_door.mojo).
    # 3. A class neither form can describe stays exactly as before its mark and is listed in
    #    context['family_msel_member_unregistered'].
    import sys
    import inspect
    scanned = context.setdefault('family_msel_member_scanned', set())
    waiting = context.setdefault('family_msel_member_unregistered', [])
    waiting[:] = [cls for cls in waiting if cls not in registry]  # glue: classes a later schema registered
    prefix = __package__ + '.'
    for name, module in tuple(sys.modules.items()):  # glue: loaded package modules, each scanned once
        if module is None or not name.startswith(prefix) or name in scanned:
            continue
        fields = vars(module)
        marked = [value for value in fields.values()  # glue: import-time class metadata of one module
                  if type(value) is type and vars(value).get('__module__') == name
                  and type(vars(value).get('_mojolearn_member')) is tuple]
        rows = [_MEMBER_SCHEMAS.get(vars(cls)['_mojolearn_member'][0]) for cls in marked]  # glue: one family row per marked class
        if any(row is not None and (row[0] not in fields or row[1] not in fields) for row in rows):  # glue: owner names of one module
            continue  # the module's owner is not connected yet (still importing): scan it at the next connect
        scanned.add(name)
        for cls in marked:  # glue: one fixed schema per marked class
            if cls in registry:
                continue
            row = _MEMBER_SCHEMAS.get(vars(cls)['_mojolearn_member'][0])
            names = getattr(cls, '_parameters', None) or tuple(inspect.signature(cls).parameters)
            if row is not None:
                methods = row[2](cls)
                register(cls, fields[row[0]], fields[row[1]], names, row[3] if len(row) > 3 else None, methods)
                plan = registry[cls]
            else:
                derived = _derive_member(cls, fields, names)
                if derived is None:
                    if cls not in waiting:
                        waiting.append(cls)
                    continue
                methods = {}
                register(cls, derived[0], derived[1], names, None, methods)
                plan = registry[cls]
                plan['member_only'] = True
            plan['initial_fields'] = {} if row is not None else derived[2]
            plan['keywords'] = {key: () for key in methods}  # glue: no keyword arguments in these recipes
    _connect_host_members(registry, register, waiting)
    _connect_search_members(context, registry, register, waiting)
    _connect_feature_freq_members(registry, register, waiting)
    _connect_host_tree_members(registry, register, waiting)


#: 6. The Gbdt member kinds with a native constructor `_derive_member` cannot read (its body calls the owner,
#: and the constructor stores the booster's settings beside its parameters), by mark: the constructor's
#: parameters in its body's order, that native constructor operation, and the keywords its fit takes beyond
#: X and y. ExperimentalTwoLevelFeatureFreq (16, 3; lane mf-meta5); OrderedRMSE (16, 4; lane mf-hostent5: its
#: required `permutation` and its sample_weight reach the member from a caller's fit parameters).
_TEXT_MODEL_MEMBERS = {
    (16, 3): (('sources', 'learning_rate', 'l2_leaf_reg', 'random_state', 'numeric_mode', 'device'),
              'trees_gbdt_feature_freq_init', ()),
    (16, 4): (('n_estimators', 'max_depth', 'learning_rate', 'l2_leaf_reg', 'border_count', 'numeric_mode',
               'device'), 'trees_gbdt_ordered_init', ('permutation', 'sample_weight')),
    # GradientBoosting itself, the booster (16, 1; lane w3-fix-a, row 2): its constructor's parameters are the
    # module's `_INIT_NAMES` (the `trees_gbdt_init` order, `device` last) then `numeric_mode` (None here: read
    # from the module at connect, never copied); a clone passes the stored canonical values, as the set_params
    # rebuild does. Its fit keywords are its fit's (python/mojolearn/ensemble.py `GradientBoosting.fit`). Before,
    # no plan was registered and RFE refused it ("RFE requires a mojolearn estimator; got GradientBoosting").
    (16, 1): (None, 'trees_gbdt_init',
              ('sample_weight', 'eval_set', 'group_id', 'subgroup_id', 'pairs', 'pairs_weight')),
}


def _connect_feature_freq_members(registry, register, waiting):
    # glue: import-time class metadata. 6. A class of `_TEXT_MODEL_MEMBERS` is registered as a MEMBER-ONLY plan
    # with its constructor's parameters and native constructor operation, so a clone is built by the
    # constructor, as `_connect_native_init` registers it.
    for cls in tuple(waiting):  # glue: the classes the scan above left waiting, once per connect
        own = vars(cls)
        row = _TEXT_MODEL_MEMBERS.get(own.get('_mojolearn_member'))
        if row is None:
            continue
        module = sys.modules.get(own.get('__module__'))
        fields = vars(module) if module is not None else {}
        if '_TREES' not in fields or type(fields.get('_TREES_CONTEXT')) is not dict:
            continue
        names = row[0]
        if names is None:
            if type(fields.get('_INIT_NAMES')) is not tuple:
                continue
            names = fields['_INIT_NAMES'] + ('numeric_mode',)
        register(cls, fields['_TREES'], fields['_TREES_CONTEXT'], names, row[1], {})
        plan = registry[cls]
        plan['member_only'] = True
        plan['initial_fields'] = {}
        plan['keywords'] = {'fit': row[2]} if row[2] else {}
        waiting.remove(cls)


def _connect_host_tree_members(registry, register, waiting):
    # glue: import-time class metadata. 7. The saved-model classes HostForest (mark (14, 2)) and HostGBDT (mark
    # (16, 5)) (lane mf-hostent5): PREDICT-ONLY member kinds (core/forest_member_run.mojo, core/gbdt_member.mojo),
    # whose fitted state is imported from their own fields (`_mojolearn_member_predict_only`). They have no fit
    # and a constructor that takes the saved arrays, so `_derive_member` cannot read them: registered as
    # MEMBER-ONLY plans with no constructor parameters (a clone is never built: nothing fits them) and the tree
    # owner's entry and context (the trees bridge's, which every tree class shares), no method recipes.
    for cls in tuple(waiting):  # glue: the classes the scan above left waiting, once per connect
        if _register_host_tree(cls, registry, register):
            waiting.remove(cls)


def _register_host_tree(cls, registry, register):
    # glue: import-time class metadata of one saved-model class; True when `cls` is a predict-only host tree
    # class and is registered (now or before).
    own = vars(cls)
    if own.get('_mojolearn_member_predict_only') is not True:
        return False
    if own.get('_mojolearn_member') not in ((14, 2), (16, 5)):
        return False
    if cls in registry:
        return True
    from . import _native_trees_bridge
    entry, trees = _native_trees_bridge.connect()
    register(cls, entry, trees, (), None, {})
    plan = registry[cls]
    plan['member_only'] = True
    plan['initial_fields'] = {}
    plan['keywords'] = {}
    return True


def connect_host_tree_member(cls):
    # glue: called ONCE, at the end of the import of python/mojolearn/_forest_host.py and _gbdt_host.py (lane
    # mf-cal5). Those modules load lazily, on first touch of HostForest / HostGBDT, which can be after the model
    # selection hub connected its marked classes: the class is registered here, at its own import, with the
    # register function and registry the hub's connection kept (python/mojolearn/_native_msel_dispatch.py
    # `family_msel_register`). Before the hub exists nothing is done: its first connection scans this module.
    selection = sys.modules.get(__package__ + '.model_selection')
    context = vars(selection).get('_NATIVE_CONTEXT') if selection is not None else None
    if type(context) is not dict:
        return
    register = context.get('family_msel_register')
    registry = context.get('family_msel_estimators')
    if register is None or type(registry) is not dict:
        return
    if not _register_host_tree(cls, registry, register):
        return
    # The DeviceRows identities every plan's owner context carries (_native_msel_dispatch.connect, after
    # connect_members): the hub's loop ran before this class existed.
    from . import _arena_io
    plan_context = registry[cls]['context']
    if type(plan_context) is dict:
        plan_context.update(door_rows_class=_arena_io.DeviceRows, door_rows_context=_arena_io._ROWS_CONTEXT)
    waiting = context.get('family_msel_member_unregistered')
    if type(waiting) is list and cls in waiting:
        waiting.remove(cls)


import sys


def _connect_search_members(context, registry, register, waiting):
    # glue: import-time class metadata. 5. A marked SEARCH class (GridSearchCV, RandomizedSearchCV: the search META
    # member kind, core/ensemble_member.mojo META_SEARCH_*; lane mf-search4, 2026-11-23) has a native constructor that
    # `_derive_member` cannot read, and its storage layout is the one model selection connected at import
    # (python/mojolearn/_native_msel_search.py `family_msel_search_layouts`): it is registered as a MEMBER-ONLY plan
    # with those constructor parameters, the owner entry and context of its own module, no method recipes. A class
    # whose layout is not connected yet stays waiting and is retried at the next connect.
    layouts = context.get('family_msel_search_layouts', {})
    for cls in tuple(waiting):  # glue: the classes the scan above left waiting, once per connect
        own = vars(cls)
        layout = layouts.get(cls)
        mark = own.get('_mojolearn_member')
        if layout is None or type(mark) is not tuple or mark[0] != 39:
            continue
        module = sys.modules.get(own.get('__module__'))
        fields = vars(module) if module is not None else {}
        if '_NATIVE' not in fields or type(fields.get('_NATIVE_CONTEXT')) is not dict:
            continue
        register(cls, fields['_NATIVE'], fields['_NATIVE_CONTEXT'], layout['parameters'], None, {})
        plan = registry[cls]
        plan['member_only'] = True
        plan['initial_fields'] = {}
        plan['keywords'] = {}
        waiting.remove(cls)


def _connect_host_members(registry, register, waiting):
    # glue: import-time class metadata. 4. A marked class with the HOST-ONLY word (`_mojolearn_member_host`: the
    # explicit Host* classes of python/mojolearn/_classical_host.py, whose own fit binds the host image) has no fit
    # door of its own module, so `_derive_member` cannot read one; it is registered as a MEMBER-ONLY plan from the
    # plan of its nearest registered ancestor with the SAME mark (its public class: the same owner entry, context,
    # constructor parameters and initial fields), and the family's admit reads the host word
    # (bindings/member_owner_unpack.mojo `member_host_only`). Lane mf-cal3b, 2026-11-23. A class whose ancestor
    # is not registered yet stays waiting and is retried at the next connect.
    for cls in tuple(waiting):  # glue: the classes the scan above left waiting, once per connect
        own = vars(cls)
        if own.get('_mojolearn_member_host') is not True:
            continue
        base = next((parent for parent in cls.__mro__[1:]  # glue: the class's own ancestry, import-time
                     if parent in registry and vars(parent).get('_mojolearn_member') == own['_mojolearn_member']),
                    None)
        if base is None:
            continue
        parent = registry[base]
        register(cls, parent['entry'], parent['context'], parent['parameters'], parent['init'], {})
        plan = registry[cls]
        plan['member_only'] = True
        plan['host_only'] = True
        plan['initial_fields'] = dict(parent.get('initial_fields', {}))
        plan['keywords'] = {}
        waiting.remove(cls)
