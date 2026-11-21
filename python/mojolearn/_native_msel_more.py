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
        owner = dict(source)
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
    _connect_members(context, register, registry)


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


# Member family number (core/member_families.mojo) -> the names of the family owner's entry and context in the
# module that defines the class, and its method recipes. One row per FAMILY whose classes had no schema of their
# own; the CLASSES come from their marks (`_mojolearn_member`), never from a list here. Every constructor of
# these families only stores its parameters under their own names (init None: the native clone copies them).
_MEMBER_SCHEMAS = {
    19: ('_XSEQ', '_XSEQ_CONTEXT', _mlp_methods),
    22: ('_PLS_CALL', '_PLS_CONTEXT', _pls_methods),
}


def _connect_members(context, register, registry):
    # glue: every class of a LOADED package module that carries its OWN member mark and has no schema yet is
    # registered from its family's row, once per module (a family that loads later is scanned when it connects).
    # A marked class of a family without a row stays as it was and is listed in
    # context['family_msel_member_unregistered'] (its mark is not read until it has a schema).
    import sys
    import inspect
    scanned = context.setdefault('family_msel_member_scanned', set())
    waiting = context.setdefault('family_msel_member_unregistered', [])
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
            if row is None:
                if cls not in waiting:
                    waiting.append(cls)
                continue
            methods = row[2](cls)
            names = getattr(cls, '_parameters', None) or tuple(inspect.signature(cls).parameters)
            register(cls, fields[row[0]], fields[row[1]], names, None, methods)
            plan = registry[cls]
            plan['initial_fields'] = {}
            plan['keywords'] = {key: () for key in methods}  # glue: no keyword arguments in these recipes
