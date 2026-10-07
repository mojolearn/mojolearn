"""Testing-only capture of reviewed fitted storage after the scored operation.

No fitting, prediction, derived state or runtime fallback is performed here.
Private names are version-pinned native-result storage, never opaque handles.
Constructor settings and every retained fitted field used by prediction,
transform or supported partial_fit are retained together. This is not a new
product save API, a generic object serializer, or a claim of runtime compliance.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import sys

LINEAR = 'mojolearn._expansion_linear'
PREP = 'mojolearn._expansion_prep'
SOURCES = {
    LINEAR: dict(path='python/mojolearn/_expansion_linear.py', sha256='06d0f8d75d727afcaa3fa7d820741c9c23ab6fd3a2ccc59290303d5f14f30957'),
    PREP: dict(path='python/mojolearn/_expansion_prep.py', sha256='57e0e4ee3fd025ece2366d33d8faf70ecbbd996a2da771d1e8e7fba749259c6e'),
}

CONTRACTS = {
    'RidgeCV': (LINEAR, ('coef_', 'intercept_', 'alpha_', 'best_score_', 'n_features_in_'), ('cv_results_',)),
    'LassoCV': (LINEAR, ('coef_', 'intercept_', 'alpha_', 'n_iter_', 'alphas_', 'mse_path_', 'n_features_in_'), ()),
    'ElasticNetCV': (LINEAR, ('coef_', 'intercept_', 'alpha_', 'l1_ratio_', 'n_iter_', 'alphas_', 'mse_path_', 'n_features_in_'), ()),
    'GaussianNB': (PREP, ('classes_', 'theta_', 'var_', 'class_count_', 'class_prior_', 'epsilon_',
                         '_const', '_raw_var', 'numeric_mode_', 'n_features_in_'), ()),
    'LinearDiscriminantAnalysis': (PREP, ('classes_', 'means_', 'priors_', 'xbar_', 'coef_', 'intercept_',
                                         '_coef', '_inter', '_max_components', 'numeric_mode_', 'n_features_in_'),
                                  ('covariance_', 'scalings_', '_scal_full', '_rank', 'explained_variance_ratio_')),
    'QuadraticDiscriminantAnalysis': (PREP, ('classes_', 'means_', 'priors_', 'rotations_', 'scalings_',
                                            '_rot', '_logc', 'numeric_mode_', 'n_features_in_'), ('covariance_',)),
    'MultinomialNB': (PREP, ('classes_', 'class_count_', 'feature_count_', 'class_log_prior_',
                            'feature_log_prob_', 'numeric_mode_', 'n_features_in_'), ()),
    'ComplementNB': (PREP, ('classes_', 'class_count_', 'feature_count_', 'class_log_prior_',
                           'feature_log_prob_', 'numeric_mode_', 'n_features_in_'), ()),
    'BernoulliNB': (PREP, ('classes_', 'class_count_', 'feature_count_', 'class_log_prior_',
                          'feature_log_prob_', '_w', '_bias', 'numeric_mode_', 'n_features_in_'), ()),
    'CategoricalNB': (PREP, ('classes_', 'class_count_', 'class_log_prior_', 'category_count_',
                            'n_categories_', 'feature_log_prob_', '_cc', '_flp', '_cmax',
                            'numeric_mode_', 'n_features_in_'), ()),
    'VarianceThreshold': (PREP, ('variances_', '_mask', 'numeric_mode_', 'n_features_in_'), ()),
    'SelectKBest': (PREP, ('scores_', 'pvalues_', '_mask', 'numeric_mode_', 'n_features_in_'), ()),
}


def _copy(value):
    """Snapshot native-returned host storage; never compute missing state."""
    import numpy as np
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError('State mapping keys must be strings')
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, np.generic)) or (type(value).__module__ == 'mojolearn._array' and type(value).__name__ == 'Array'):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError('Opaque/object-valued fitted state is not qualified')
        return array.copy(order='C')
    if inspect.isfunction(value) and value.__module__ == PREP and value.__name__ in ('f_classif', 'f_regression', 'chi2'):
        return {'native_score_function': value.__module__ + '.' + value.__name__}
    raise ValueError('Unsupported retained state type: ' + type(value).__module__ + '.' + type(value).__name__)


def targeted_fitted_state(model):
    """Return complete reviewed storage or explicit UNAVAILABLE, never partial."""
    name = type(model).__name__
    if name not in CONTRACTS or type(model).__module__ != CONTRACTS[name][0]:
        return None, None
    module_name, required, optional = CONTRACTS[name]
    contract = 'mojolearn.reviewed-fitted-storage/' + name + '-1'
    metadata = dict(contract=contract, capture_source='reviewed public and private fitted native-result storage',
                    model_class=module_name + '.' + name, reviewed_api_source=SOURCES[module_name],
                    required_fields=list(required), optional_fields=list(optional),
                    excluded_non_fitted_state={'temporary_program_and_solver_workspaces': 'Released before fit returns; no persistent device handle or resumption API in reviewed classes',
                                              'vendor_and_binding_diagnostics': 'Captured separately by worker artifact/runtime provenance'},
                    numerical_verification=False, product_python_compliance='not assessed; existing timed product work unchanged')
    try:
        source = getattr(sys.modules.get(module_name), '__file__', None)
        if not source or hashlib.sha256(Path(source).read_bytes()).hexdigest() != SOURCES[module_name]['sha256']:
            raise ValueError('Installed estimator source differs from exact reviewed contract')
        # Constructor state affects future supported operations (e.g. NB
        # partial_fit and selector transform); it must not be silently omitted.
        params = {key: _copy(getattr(model, key)) for key in inspect.signature(type(model).__init__).parameters if key != 'self'}
        fields = {key: _copy(getattr(model, key)) for key in required}
        fields.update({key: _copy(getattr(model, key)) if hasattr(model, key) else None for key in optional})
        if type(fields['n_features_in_']) is not int or fields['n_features_in_'] < 1:
            raise ValueError('Missing valid fitted feature count')
        import numpy as np
        nf = fields['n_features_in_']
        def array_shape(key, shape, dtype='<f4'):
            value = fields[key]
            if not isinstance(value, np.ndarray) or value.dtype.str != dtype or value.shape != shape:
                raise ValueError(key + ': fitted dtype/shape differs from reviewed contract')
        if module_name == LINEAR:
            array_shape('coef_', (nf,))
            if name != 'RidgeCV':
                alphas, mse = fields['alphas_'], fields['mse_path_']
                if (not isinstance(alphas, np.ndarray) or not isinstance(mse, np.ndarray)
                        or alphas.dtype.str != '<f4' or mse.dtype.str != '<f4'
                        or mse.shape[:-1] != alphas.shape or mse.shape[-1] != (5 if model.cv is None else model.cv)):
                    raise ValueError('Complete CV alpha grid/fold diagnostics differ from contract')
        if 'classes_' in fields:
            nc = len(fields['classes_'])
            if nc < 1:
                raise ValueError('Missing fitted classes')
            if name == 'GaussianNB':
                for key in ('theta_', 'var_', '_raw_var'):
                    array_shape(key, (nc, nf))
                for key in ('class_count_', 'class_prior_', '_const'):
                    array_shape(key, (nc,))
            if name == 'LinearDiscriminantAnalysis':
                for key in ('means_', '_coef'):
                    array_shape(key, (nc, nf))
                array_shape('_inter', (nc,))
                array_shape('priors_', (nc,))
                array_shape('xbar_', (nf,))
                array_shape('coef_', (1 if nc == 2 else nc, nf))
                array_shape('intercept_', (1 if nc == 2 else nc,))
        if name in ('LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis') and model.covariance_estimator is not None:
            raise ValueError('External covariance estimator fitted state is outside this reviewed contract')
        if name == 'RidgeCV' and model.store_cv_results and fields['cv_results_'] is None:
            raise ValueError('Requested CV results were not retained')
        if name == 'LinearDiscriminantAnalysis':
            if model.solver not in ('svd', 'lsqr', 'eigen'):
                raise ValueError('Unreviewed LDA solver')
            if model.solver != 'lsqr' and any(fields[key] is None for key in ('scalings_', '_scal_full', '_rank', 'explained_variance_ratio_')):
                raise ValueError('LDA transform fitted state is incomplete')
            if (model.store_covariance or model.solver != 'svd') and fields['covariance_'] is None:
                raise ValueError('LDA covariance fitted state is incomplete')
        if name == 'QuadraticDiscriminantAnalysis' and model.store_covariance and fields['covariance_'] is None:
            raise ValueError('QDA covariance fitted state is incomplete')
        from six_lane_evidence import value_manifest
        state = dict(estimator=module_name + '.' + name, format=contract, fitted=fields, constructor=params)
        metadata['contract_paths'] = [row['path'] for row in value_manifest(state)]
        metadata['contract_source'] = SOURCES[module_name]['path']
        return state, metadata
    except Exception as exc:
        return None, dict(metadata, status='UNAVAILABLE', completeness='incomplete',
                          missing_state=['complete reviewed fitted storage'], reason=type(exc).__name__ + ': ' + str(exc))
