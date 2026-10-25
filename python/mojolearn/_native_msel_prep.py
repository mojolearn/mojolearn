# SPDX-License-Identifier: Apache-2.0
"""Import-time native plans for estimators composed by RFE and IterativeImputer."""


def connect(context, register, registry):
    # glue: fixed owner ABIs and constructor metadata, resolved once at import.
    # No signature inspection, binding lookup or Python constructor runs inside
    # a composed estimator operation.
    import inspect
    from . import _backend, _expansion_linear as linear, _solver_impl as cd
    from . import _svm_impl as svm, svm as linear_svm, _gp_impl as gp
    from . import kernel_methods as km, neighbors, linear_model, _expansion_trees as trees

    arg = lambda n: ('arg', n)
    constant = lambda value: ('constant', value)
    keyword = lambda name, default=None: ('keyword', name, default)
    binding = lambda name: ('binding', name)

    def add(cls, entry, owner, methods, *, initial=None, parameters=None):
        # glue: constructor names are copied from the exact built-in class.
        names = tuple(inspect.signature(cls).parameters) if parameters is None else parameters
        register(cls, entry, owner, names, None, methods)
        plan = registry[cls]
        plan['initial_fields'] = {} if initial is None else initial
        plan['keywords'] = {
            name: tuple(item[1] for item in recipe if item[0] == 'keyword')  # glue: fixed ABI metadata connected once during package import
            for name, (_, recipe) in methods.items()  # glue: fixed ABI metadata connected once during package import
        }
        return plan

    # Restore every linear expansion that accepts a matrix and exposes a
    # supervised fit. IsotonicRegression accepts a vector, so was never a
    # supported RFE/IterativeImputer estimator; SGDOneClassSVM has no target.
    for name, fit, classifier, weighted in (  # glue: fixed ABI metadata connected once during package import
        ('SGDClassifier', 'fit_sgd_classifier', True, True),
        ('SGDRegressor', 'fit_sgd_regressor', False, True),
        ('PoissonRegressor', 'fit_glm', False, True),
        ('GammaRegressor', 'fit_glm', False, True),
        ('TweedieRegressor', 'fit_glm', False, True),
        ('HuberRegressor', 'fit_huber', False, True),
        ('BayesianRidge', 'fit_bayes', False, True),
        ('ARDRegression', 'fit_ard', False, False),
        ('Lars', 'fit_lars', False, False),
        ('LassoLars', 'fit_lasso_lars', False, False),
        ('QuantileRegressor', 'fit_quantile', False, True),
        ('Perceptron', 'fit_perceptron', True, True),
        ('PassiveAggressiveClassifier', 'fit_pa_classifier', True, False),
        ('PassiveAggressiveRegressor', 'fit_pa_regressor', False, False),
        ('RidgeClassifier', 'fit_ridge_classifier', True, True),
        ('RidgeCV', 'fit_ridge_cv', False, True),
        ('LassoCV', 'fit_lasso_cv', False, True),
        ('ElasticNetCV', 'fit_elastic_net_cv', False, True),
        ('LogisticRegressionCV', 'fit_logistic_cv', True, True),
    ):
        methods = {
            'fit': (fit, (arg(0), arg(1)) + ((keyword('sample_weight'),) if weighted else ())),
            'predict': ('classifier_predict' if classifier else 'regressor_predict', (arg(0),)),
            'score': ('classifier_score' if classifier else 'linear_regressor_score', (arg(0), arg(1))),
        }
        if classifier:
            methods['decision_function'] = ('decision_function', (arg(0),))
        if name in ('SGDClassifier', 'LogisticRegressionCV'):
            methods['predict_proba'] = ('sgd_predict_proba' if name == 'SGDClassifier' else 'lrcv_predict_proba', (arg(0),))
        plan = add(getattr(linear, name), linear._XL_CALL, linear._XL_CONTEXT, methods)
        if not classifier:
            plan['method_entries'] = {'score': (linear._LINEAR_SCORE, linear._LINEAR_SCORE_CONTEXT)}

    for cls in (cd.ElasticNet, cd.Lasso):  # glue: fixed ABI metadata connected once during package import
        add(cls, cd._CD_CALL, cd._CD_CONTEXT, {
            'fit': ('cd_fit', (arg(0), arg(1), keyword('sample_weight'))),
            'predict': ('cd_predict', (arg(0),)),
            'score': ('cd_score', (arg(0), arg(1))),
        }, initial={'l1_ratio': 1.0} if cls is cd.Lasso else None)

    add(linear_model.QNRegressor, linear_model._QN_CALL, linear_model._QN_CONTEXT, {
        'fit': ('fit', (arg(0), arg(1), keyword('sample_weight'))),
        'predict': ('predict', (arg(0),)),
        'score': ('score', (arg(0), arg(1))),
    })

    for name in ('BaggingClassifier', 'BaggingRegressor', 'AdaBoostClassifier', 'AdaBoostRegressor',  # glue: fixed ABI metadata connected once during package import
                 'DARTClassifier', 'DARTRegressor'):
        methods = {
            'fit': ('trees_boost_fit', (arg(0), arg(1), constant(None) if name.startswith('DART') else keyword('sample_weight'))),
            'predict': ('trees_boost_predict', (arg(0),)),
            'score': ('trees_boost_score', (arg(0), arg(1), constant(None))),
        }
        if name.endswith('Classifier'):
            methods['predict_proba'] = ('trees_boost_predict_proba', (arg(0),))
        if name == 'AdaBoostClassifier':
            methods['decision_function'] = ('trees_boost_decision_function', (arg(0),))
        add(getattr(trees, name), trees._TREES, trees._TREES_CONTEXT, methods)

    for cls, prefix in ((svm.SVC, 'svc'), (svm.SVR, 'svr')):  # glue: fixed ABI metadata connected once during package import
        native = binding('_mojolearn_svm')
        methods = {
            'fit': ('family_svc_' + prefix + '_fit', (native, arg(0), arg(1), keyword('sample_weight'))),
            'predict': ('family_svc_' + prefix + '_predict', (native, arg(0))),
            'score': ('family_svc_' + prefix + '_score', (native, arg(0), arg(1))),
            'coef_': ('family_svc_' + prefix + '_coef', (native,)),
        }
        if cls is svm.SVC:
            methods.update(decision_function=('family_svc_svc_decision', (native, arg(0))),
                           predict_proba=('family_svc_svc_proba', (native, arg(0))))
        add(cls, svm._NATIVE, svm._NATIVE_CONTEXT, methods)

    for cls, prefix in ((linear_svm.LinearSVC, 'svc'), (linear_svm.LinearSVR, 'svr')):  # glue: fixed ABI metadata connected once during package import
        native = binding('_mojolearn_estimators')
        methods = {
            'fit': ('family_svc_linear_' + prefix + '_fit', (native, arg(0), arg(1), keyword('sample_weight'))),
            'predict': ('family_svc_linear_svc_predict' if prefix == 'svc' else 'family_svm_linear_svr_predict', (native, arg(0))),
            'score': ('family_svc_linear_' + prefix + '_score', (native, arg(0), arg(1))),
        }
        if prefix == 'svc':
            methods['decision_function'] = ('family_svm_linear_svc_decision', (native, arg(0)))
        add(cls, linear_svm._NATIVE, linear_svm._NATIVE_CONTEXT, methods)

    native = binding('_mojolearn_gp')
    add(gp.GaussianProcessRegressor, gp._NATIVE, gp._NATIVE_CONTEXT, {
        'fit': ('family_gp_gpr_fit', (native, arg(0), arg(1))),
        'predict': ('family_gp_gpr_predict', (native, arg(0), constant(False), constant(False))),
        'predict_std': ('family_gp_gpr_predict', (native, arg(0), constant(True), constant(False))),
        'score': ('family_gp_gpr_score', (native, arg(0), arg(1))),
    })
    # Kernel objects are immutable constructor specifications copied through
    # native instance dictionaries, never _nodes/_replace/deepcopy callbacks.
    for cls, names in (  # glue: fixed ABI metadata connected once during package import
        (gp.ConstantKernel, ('constant_value', 'constant_value_bounds')),
        (gp.WhiteKernel, ('noise_level', 'noise_level_bounds')),
        (gp.RBF, ('length_scale', 'length_scale_bounds')),
        (gp.Matern, ('length_scale', 'length_scale_bounds', 'nu')),
        (gp._Combined, ('a', 'b', 'op', 'sym')),
        (gp._SavedKernel, ('_SavedKernel__nodes',)),
    ):
        add(cls, gp._NATIVE, gp._NATIVE_CONTEXT, {}, parameters=names)

    native = binding('_mojolearn_kernel_methods')
    add(km.KernelRidge, km._NATIVE, km._NATIVE_CONTEXT, {
        'fit': ('family_km_krr_fit', (native, arg(0), arg(1), keyword('sample_weight'))),
        'predict': ('family_km_krr_predict', (native, arg(0))),
        'score': ('family_km_krr_score', (native, arg(0), arg(1), constant(None), binding('_mojolearn_metrics'))),
    })
    for cls, prefix in ((neighbors.KNeighborsClassifier, 'clf'), (neighbors.KNeighborsRegressor, 'reg')):  # glue: fixed ABI metadata connected once during package import
        methods = {
            'fit': ('family_xn_knn_' + prefix + '_fit', (binding('_mojolearn_x_neighbors'), arg(0), arg(1))),
            'predict': ('family_xn_knn_' + prefix + '_predict', (binding('_mojolearn'), arg(0))),
            'score': ('family_xn_knn_' + prefix + '_score', (binding('_mojolearn'), arg(0), arg(1), constant(None))),
        }
        if prefix == 'clf':
            methods['predict_proba'] = ('family_xn_knn_clf_predict_proba', (binding('_mojolearn'), arg(0)))
        add(cls, neighbors._NATIVE, neighbors._NATIVE_CONTEXT, methods,
            initial={'_index': None, '_y_cols': None})

    # Existing plans used to discard these fit parameters unconditionally.
    # Match each public fit's ABI, including forest importances and boost eval.
    for cls, plan in tuple(registry.items()):  # glue: fixed ABI metadata connected once during package import
        name = cls.__name__.removeprefix('Host')
        params = {'sample_weight': (2, None)} if name in (
            'LinearRegression', 'Ridge', 'LogisticRegression', 'GaussianNB',
            'RandomForestClassifier', 'RandomForestRegressor', 'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor', 'GradientBoostingClassifier', 'GradientBoostingRegressor',
        ) else {}
        if name in ('RandomForestClassifier', 'RandomForestRegressor', 'ExtraTreesClassifier', 'ExtraTreesRegressor'):
            params['feature_importances'] = (3, False)
        if 'fit_entry' in plan:
            plan.setdefault('keywords', {})['fit'] = ('sample_weight',)
        if name in ('GradientBoostingClassifier', 'GradientBoostingRegressor'):
            params['eval_set'] = (3, None)
        if not params:
            continue
        operation, old = plan['methods']['fit']
        recipe = list(old)
        for key, (position, default) in params.items():  # glue: fixed ABI metadata connected once during package import
            recipe[position] = keyword(key, default)
        plan['methods']['fit'] = (operation, tuple(recipe))
        plan.setdefault('keywords', {})['fit'] = tuple(params)

    wanted = {item[1] for plan in registry.values() for _, recipe in plan['methods'].values()  # glue: fixed ABI metadata connected once during package import
              for item in recipe if item[0] == 'binding'}  # glue: fixed ABI metadata connected once during package import
    modules = {}
    for basename in sorted(wanted):  # glue: fixed ABI metadata connected once during package import
        modes = {}
        for mode in ('identical', 'fast'):  # glue: fixed ABI metadata connected once during package import
            try:
                modes[mode] = _backend.binding(basename, mode)
            except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
                modes[mode] = type(error).__name__ + ': ' + str(error)
        modules[basename] = modes
    context['family_msel_bindings'] = modules
