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
    names = ('priors', 'var_smoothing')
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
