# SPDX-License-Identifier: Apache-2.0
"""Import-time constructor plans for the native linear parameter setter."""
from types import BuiltinFunctionType
from .tokenizer import _binding

_ENTRY = _binding()._module.params_setter_native_call
if type(_ENTRY) is not BuiltinFunctionType:
    raise TypeError('parameter setters require a compiled native entry')
_CONTEXT = dict(plans={}, missing=object(), ready=False)

# These are the field-only constructors already connected by
# _native_msel_prep. No clone callback or parameter copy runs in set_params.
_CLASSES = (
    'SGDClassifier', 'SGDRegressor', 'PoissonRegressor', 'GammaRegressor',
    'TweedieRegressor', 'HuberRegressor', 'BayesianRidge', 'ARDRegression',
    'Lars', 'LassoLars', 'QuantileRegressor', 'Perceptron',
    'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor',
    'RidgeClassifier', 'RidgeCV', 'LassoCV', 'ElasticNetCV', 'LogisticRegressionCV',
)

# Union of the linear native owner's fitted writes, plus shared fitted-state
# bookkeeping. Do not erase arbitrary user fields, including names ending '_'.
_FITTED_FIELDS = (
    'C_', 'Cs_', 'X_max_', 'X_min_', 'X_thresholds_', '_link', 'active_',
    'alpha_', 'alphas_', 'best_score_', 'classes_', 'coef_', 'cv_results_',
    'fortran_copied_', 'increasing_', 'input_copied_', 'intercept_', 'l1_ratio_',
    'lambda_', 'mse_path_', 'n_features_in_', 'n_iter_', 'offset_', 'outliers_',
    'scale_', 'scores_', 't_', 'y_thresholds_', 'numeric_mode_',
    'feature_names_in_', '_mojolearn_fitted', '_mojolearn_fitting', '_mojolearn_params_reset',
)


class NativeParamsSetter:
    def set_params(self, **params):
        """Set constructor parameters and clear fitted state.

        Parameters
        ----------
        **params : keyword arguments
            Constructor parameters; use ``name__parameter`` for nested values.

        Returns
        -------
        self : object
            This estimator with its updated parameters.
        """
        return _ENTRY(self, params, _CONTEXT)


def connect(context, parameter_metadata):
    # glue: exact built-in constructor and native storage identities at import
    from types import ModuleType
    from . import _expansion_linear as linear
    from ._native_classical_bridge import _numpy_free_context
    from ._native_owner_lifetime import connect_owner_context
    ownership = _numpy_free_context()
    ownership['native_module_type'] = ModuleType
    connect_owner_context(ownership)
    plans = {}
    for name in _CLASSES:  # glue: fixed native field-assignment constructor schemas
        cls = vars(linear)[name]
        plan = context['family_msel_estimators'][cls]
        if plan['init'] is not None or plan['forest'] is not None or plan.get('initial_fields'):
            raise TypeError('linear parameter setter requires its field-only native constructor plan')
        plans[cls] = dict(constructor=plan, class_name=name, fitted_fields=_FITTED_FIELDS)
    # Existing CV configurations can be replaced without executing their
    # constructors or lifecycle hooks. This does not add a splitter setter.
    splitters = {cls: context['family_msel_estimators'][cls]  # glue: exact previously connected native splitter ownership plans
                 for cls in context['family_msel_splitter_layouts']}
    _CONTEXT.update(plans=plans, parameter_metadata=parameter_metadata,
                    ownership=ownership, splitter_plans=splitters, ready=True)
