# SPDX-License-Identifier: Apache-2.0
"""Additional fixed estimator schemas for native model selection."""


def connect(context, register, registry):
    # glue: called once after package classes and their compiled doors exist
    from . import _expansion_prep as prep
    from ._native_classical_bridge import _layout
    cls = prep.LinearDiscriminantAnalysis
    parameters = ('solver', 'shrinkage', 'priors', 'n_components', 'store_covariance', 'tol',
                  'covariance_estimator')
    methods = {
        'fit': ('fit', (('arg', 0), ('arg', 1))),
        'predict': ('predict', (('arg', 0),)),
        'predict_proba': ('predict_proba', (('arg', 0),)),
        'predict_log_proba': ('predict_log_proba', (('arg', 0),)),
        'decision_function': ('decision_function', (('arg', 0),)),
        'transform': ('transform', (('arg', 0),)),
        'score': ('score', (('arg', 0), ('arg', 1))),
        'get_params': ('get_params', (('constant', False),)),
        'set_params': ('set_params', (('arg', 0),)),
    }
    # __init__ assigns these seven objects unchanged. Native cloning copies
    # their supported values and writes the fresh instance dictionary; no
    # Python constructor or ParamsMixin.get_params is an execution entry.
    register(cls, prep._CLS_CALL, prep._CLS_CONTEXT, parameters, 'initialize', methods)
    layout = _layout(cls, None)
    prep._CLS_CONTEXT.update(
        lda_native_class=cls, lda_parameters=parameters,
        lda_admission_context={'classical_public_layouts': {cls: layout}, 'classical_public_installed': True},
        lda_fields_get=layout['fields_get'],
        lda_fitted_fields=('classes_', 'n_features_in_', 'numeric_mode_', 'coef_', 'intercept_', '_coef', '_inter',
                           'means_', 'priors_', 'xbar_', 'scalings_', 'explained_variance_ratio_', 'covariance_',
                           '_rank', '_max_components', '_scal_full'),
    )
