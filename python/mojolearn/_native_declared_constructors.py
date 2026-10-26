# SPDX-License-Identifier: Apache-2.0
"""Import-time schemas for the field-only constructors of the declared families.

The prep (`_PrepBase`) and decomp (`_Base`) classes whose `set_params` is the
shallow declared assignment (`_native_declared_params.py`), and a few plain
classes (the shuffle and leave-P splitters, two resample results), have
constructors that only store their parameters. Their public `__init__` is one call to the
same native constructor as the full-reset owners
(bindings/full_reset_constructor_native.mojo); re-initialization releases an
overwritten value under the declared setter's own admission. The write events
are derived from source by tools/declared_constructor_schemas.py and checked
against it by tools/test_native_declared_constructor_connection.py.
"""
from types import BuiltinFunctionType
from .tokenizer import _binding
from . import _expansion_prep as _prep
from . import _expansion_decomp as _decomp
from . import model_selection as _msel
from . import resample as _resample

_ENTRY = _binding()._module.full_reset_constructor_native_call
if type(_ENTRY) is not BuiltinFunctionType:
    raise TypeError('declared constructors require a compiled native entry')
_CONTEXT = dict(plans={}, missing=object(), ready=False, originals={}, release_declared=True)
_CONSTRUCTOR_HOOKS = ('__getattribute__', '__getattr__', '__setattr__')
_CLASSES = (
    _decomp.AlternatingLeastSquares,
    _prep.BernoulliNB,
    _prep.Binarizer,
    _decomp.CCA,
    _prep.CategoricalNB,
    _decomp.ClassicalMDS,
    _prep.ComplementNB,
    _decomp.DictionaryLearning,
    _decomp.EllipticEnvelope,
    _decomp.FactorAnalysis,
    _decomp.FastICA,
    _prep.GaussianNB,
    _decomp.GaussianRandomProjection,
    _msel.GroupShuffleSplit,
    _decomp.IncrementalPCA,
    _decomp.Isomap,
    _prep.IterativeImputer,
    _prep.KBinsDiscretizer,
    _prep.LabelBinarizer,
    _decomp.LatentDirichletAllocation,
    _msel.LeavePGroupsOut,
    _msel.LeavePOut,
    _decomp.LocallyLinearEmbedding,
    _decomp.MDS,
    _decomp.MinCovDet,
    _decomp.MiniBatchDictionaryLearning,
    _decomp.MiniBatchSparsePCA,
    _resample.MonteCarloResult,
    _prep.MultiLabelBinarizer,
    _prep.MultinomialNB,
    _decomp.NMF,
    _prep.Normalizer,
    _prep.OneHotEncoder,
    _prep.OrdinalEncoder,
    _decomp.PLSCanonical,
    _decomp.PLSRegression,
    _resample.PermutationTestResult,
    _prep.PolynomialFeatures,
    _prep.PowerTransformer,
    _prep.QuadraticDiscriminantAnalysis,
    _prep.QuantileTransformer,
    _prep.RFE,
    _prep.SelectKBest,
    _msel.ShuffleSplit,
    _prep.SimpleImputer,
    _decomp.SparseCoder,
    _decomp.SparsePCA,
    _decomp.SparseRandomProjection,
    _prep.SplineTransformer,
    _msel.StratifiedShuffleSplit,
    _prep.TargetEncoder,
    _prep.VarianceThreshold,
)
# Constructor write events: field, parameter/constant, value/name, mode rule
# (always False: none of these classes is a NumericModeMixin).
_EVENTS = {
    'AlternatingLeastSquares': (('factors', 'parameter', 'factors', False), ('regularization', 'parameter', 'regularization', False), ('alpha', 'parameter', 'alpha', False), ('iterations', 'parameter', 'iterations', False), ('use_cg', 'parameter', 'use_cg', False), ('calculate_training_loss', 'parameter', 'calculate_training_loss', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('cg_steps', 'parameter', 'cg_steps', False)),
    'BernoulliNB': (('alpha', 'parameter', 'alpha', False), ('force_alpha', 'parameter', 'force_alpha', False), ('binarize', 'parameter', 'binarize', False), ('fit_prior', 'parameter', 'fit_prior', False), ('class_prior', 'parameter', 'class_prior', False)),
    'Binarizer': (('threshold', 'parameter', 'threshold', False), ('copy', 'parameter', 'copy', False)),
    'CCA': (('n_components', 'parameter', 'n_components', False), ('scale', 'parameter', 'scale', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('copy', 'parameter', 'copy', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'CategoricalNB': (('alpha', 'parameter', 'alpha', False), ('force_alpha', 'parameter', 'force_alpha', False), ('fit_prior', 'parameter', 'fit_prior', False), ('class_prior', 'parameter', 'class_prior', False), ('min_categories', 'parameter', 'min_categories', False)),
    'ClassicalMDS': (('n_components', 'parameter', 'n_components', False), ('metric', 'parameter', 'metric', False), ('metric_params', 'parameter', 'metric_params', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'ComplementNB': (('alpha', 'parameter', 'alpha', False), ('force_alpha', 'parameter', 'force_alpha', False), ('fit_prior', 'parameter', 'fit_prior', False), ('class_prior', 'parameter', 'class_prior', False), ('norm', 'parameter', 'norm', False)),
    'DictionaryLearning': (('n_components', 'parameter', 'n_components', False), ('alpha', 'parameter', 'alpha', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('fit_algorithm', 'parameter', 'fit_algorithm', False), ('transform_algorithm', 'parameter', 'transform_algorithm', False), ('transform_n_nonzero_coefs', 'parameter', 'transform_n_nonzero_coefs', False), ('transform_alpha', 'parameter', 'transform_alpha', False), ('n_jobs', 'parameter', 'n_jobs', False), ('code_init', 'parameter', 'code_init', False), ('dict_init', 'parameter', 'dict_init', False), ('callback', 'parameter', 'callback', False), ('verbose', 'parameter', 'verbose', False), ('split_sign', 'parameter', 'split_sign', False), ('random_state', 'parameter', 'random_state', False), ('positive_code', 'parameter', 'positive_code', False), ('positive_dict', 'parameter', 'positive_dict', False), ('transform_max_iter', 'parameter', 'transform_max_iter', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'EllipticEnvelope': (('store_precision', 'parameter', 'store_precision', False), ('assume_centered', 'parameter', 'assume_centered', False), ('support_fraction', 'parameter', 'support_fraction', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('contamination', 'parameter', 'contamination', False)),
    'FactorAnalysis': (('n_components', 'parameter', 'n_components', False), ('tol', 'parameter', 'tol', False), ('copy', 'parameter', 'copy', False), ('max_iter', 'parameter', 'max_iter', False), ('noise_variance_init', 'parameter', 'noise_variance_init', False), ('svd_method', 'parameter', 'svd_method', False), ('iterated_power', 'parameter', 'iterated_power', False), ('rotation', 'parameter', 'rotation', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'FastICA': (('n_components', 'parameter', 'n_components', False), ('algorithm', 'parameter', 'algorithm', False), ('whiten', 'parameter', 'whiten', False), ('fun', 'parameter', 'fun', False), ('fun_args', 'parameter', 'fun_args', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('w_init', 'parameter', 'w_init', False), ('whiten_solver', 'parameter', 'whiten_solver', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'GaussianNB': (('priors', 'parameter', 'priors', False), ('var_smoothing', 'parameter', 'var_smoothing', False)),
    'GaussianRandomProjection': (('n_components', 'parameter', 'n_components', False), ('eps', 'parameter', 'eps', False), ('compute_inverse_components', 'parameter', 'compute_inverse_components', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'GroupShuffleSplit': (('n_splits', 'parameter', 'n_splits', False), ('test_size', 'parameter', 'test_size', False), ('train_size', 'parameter', 'train_size', False), ('random_state', 'parameter', 'random_state', False)),
    'IncrementalPCA': (('n_components', 'parameter', 'n_components', False), ('whiten', 'parameter', 'whiten', False), ('copy', 'parameter', 'copy', False), ('batch_size', 'parameter', 'batch_size', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'Isomap': (('n_neighbors', 'parameter', 'n_neighbors', False), ('radius', 'parameter', 'radius', False), ('n_components', 'parameter', 'n_components', False), ('eigen_solver', 'parameter', 'eigen_solver', False), ('tol', 'parameter', 'tol', False), ('max_iter', 'parameter', 'max_iter', False), ('path_method', 'parameter', 'path_method', False), ('neighbors_algorithm', 'parameter', 'neighbors_algorithm', False), ('n_jobs', 'parameter', 'n_jobs', False), ('metric', 'parameter', 'metric', False), ('p', 'parameter', 'p', False), ('metric_params', 'parameter', 'metric_params', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'IterativeImputer': (('estimator', 'parameter', 'estimator', False), ('missing_values', 'parameter', 'missing_values', False), ('sample_posterior', 'parameter', 'sample_posterior', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('n_nearest_features', 'parameter', 'n_nearest_features', False), ('initial_strategy', 'parameter', 'initial_strategy', False), ('fill_value', 'parameter', 'fill_value', False), ('imputation_order', 'parameter', 'imputation_order', False), ('skip_complete', 'parameter', 'skip_complete', False), ('min_value', 'parameter', 'min_value', False), ('max_value', 'parameter', 'max_value', False), ('verbose', 'parameter', 'verbose', False), ('random_state', 'parameter', 'random_state', False), ('add_indicator', 'parameter', 'add_indicator', False), ('keep_empty_features', 'parameter', 'keep_empty_features', False)),
    'KBinsDiscretizer': (('n_bins', 'parameter', 'n_bins', False), ('encode', 'parameter', 'encode', False), ('strategy', 'parameter', 'strategy', False), ('quantile_method', 'parameter', 'quantile_method', False), ('dtype', 'parameter', 'dtype', False), ('subsample', 'parameter', 'subsample', False), ('random_state', 'parameter', 'random_state', False)),
    'LabelBinarizer': (('neg_label', 'parameter', 'neg_label', False), ('pos_label', 'parameter', 'pos_label', False), ('sparse_output', 'parameter', 'sparse_output', False)),
    'LatentDirichletAllocation': (('n_components', 'parameter', 'n_components', False), ('doc_topic_prior', 'parameter', 'doc_topic_prior', False), ('topic_word_prior', 'parameter', 'topic_word_prior', False), ('learning_method', 'parameter', 'learning_method', False), ('learning_decay', 'parameter', 'learning_decay', False), ('learning_offset', 'parameter', 'learning_offset', False), ('max_iter', 'parameter', 'max_iter', False), ('batch_size', 'parameter', 'batch_size', False), ('evaluate_every', 'parameter', 'evaluate_every', False), ('total_samples', 'parameter', 'total_samples', False), ('perp_tol', 'parameter', 'perp_tol', False), ('mean_change_tol', 'parameter', 'mean_change_tol', False), ('max_doc_update_iter', 'parameter', 'max_doc_update_iter', False), ('n_jobs', 'parameter', 'n_jobs', False), ('verbose', 'parameter', 'verbose', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'LeavePGroupsOut': (('n_groups', 'parameter', 'n_groups', False),),
    'LeavePOut': (('p', 'parameter', 'p', False),),
    'LocallyLinearEmbedding': (('n_neighbors', 'parameter', 'n_neighbors', False), ('n_components', 'parameter', 'n_components', False), ('reg', 'parameter', 'reg', False), ('eigen_solver', 'parameter', 'eigen_solver', False), ('tol', 'parameter', 'tol', False), ('max_iter', 'parameter', 'max_iter', False), ('method', 'parameter', 'method', False), ('hessian_tol', 'parameter', 'hessian_tol', False), ('modified_tol', 'parameter', 'modified_tol', False), ('neighbors_algorithm', 'parameter', 'neighbors_algorithm', False), ('random_state', 'parameter', 'random_state', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MDS': (('n_components', 'parameter', 'n_components', False), ('metric_mds', 'parameter', 'metric_mds', False), ('n_init', 'parameter', 'n_init', False), ('init', 'parameter', 'init', False), ('max_iter', 'parameter', 'max_iter', False), ('verbose', 'parameter', 'verbose', False), ('eps', 'parameter', 'eps', False), ('n_jobs', 'parameter', 'n_jobs', False), ('random_state', 'parameter', 'random_state', False), ('metric', 'parameter', 'metric', False), ('metric_params', 'parameter', 'metric_params', False), ('normalized_stress', 'parameter', 'normalized_stress', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MinCovDet': (('store_precision', 'parameter', 'store_precision', False), ('assume_centered', 'parameter', 'assume_centered', False), ('support_fraction', 'parameter', 'support_fraction', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MiniBatchDictionaryLearning': (('n_components', 'parameter', 'n_components', False), ('alpha', 'parameter', 'alpha', False), ('max_iter', 'parameter', 'max_iter', False), ('fit_algorithm', 'parameter', 'fit_algorithm', False), ('n_jobs', 'parameter', 'n_jobs', False), ('batch_size', 'parameter', 'batch_size', False), ('shuffle', 'parameter', 'shuffle', False), ('dict_init', 'parameter', 'dict_init', False), ('transform_algorithm', 'parameter', 'transform_algorithm', False), ('transform_n_nonzero_coefs', 'parameter', 'transform_n_nonzero_coefs', False), ('transform_alpha', 'parameter', 'transform_alpha', False), ('verbose', 'parameter', 'verbose', False), ('split_sign', 'parameter', 'split_sign', False), ('random_state', 'parameter', 'random_state', False), ('positive_code', 'parameter', 'positive_code', False), ('positive_dict', 'parameter', 'positive_dict', False), ('transform_max_iter', 'parameter', 'transform_max_iter', False), ('callback', 'parameter', 'callback', False), ('tol', 'parameter', 'tol', False), ('max_no_improvement', 'parameter', 'max_no_improvement', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MiniBatchSparsePCA': (('n_components', 'parameter', 'n_components', False), ('alpha', 'parameter', 'alpha', False), ('ridge_alpha', 'parameter', 'ridge_alpha', False), ('max_iter', 'parameter', 'max_iter', False), ('callback', 'parameter', 'callback', False), ('batch_size', 'parameter', 'batch_size', False), ('verbose', 'parameter', 'verbose', False), ('shuffle', 'parameter', 'shuffle', False), ('n_jobs', 'parameter', 'n_jobs', False), ('method', 'parameter', 'method', False), ('random_state', 'parameter', 'random_state', False), ('tol', 'parameter', 'tol', False), ('max_no_improvement', 'parameter', 'max_no_improvement', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MonteCarloResult': (('integral', 'parameter', 'integral', False), ('mean', 'parameter', 'mean', False), ('volume', 'parameter', 'volume', False), ('closed_form', 'parameter', 'closed_form', False)),
    'MultiLabelBinarizer': (('classes', 'parameter', 'classes', False), ('sparse_output', 'parameter', 'sparse_output', False)),
    'MultinomialNB': (('alpha', 'parameter', 'alpha', False), ('force_alpha', 'parameter', 'force_alpha', False), ('fit_prior', 'parameter', 'fit_prior', False), ('class_prior', 'parameter', 'class_prior', False)),
    'NMF': (('n_components', 'parameter', 'n_components', False), ('init', 'parameter', 'init', False), ('solver', 'parameter', 'solver', False), ('beta_loss', 'parameter', 'beta_loss', False), ('tol', 'parameter', 'tol', False), ('max_iter', 'parameter', 'max_iter', False), ('random_state', 'parameter', 'random_state', False), ('alpha_W', 'parameter', 'alpha_W', False), ('alpha_H', 'parameter', 'alpha_H', False), ('l1_ratio', 'parameter', 'l1_ratio', False), ('verbose', 'parameter', 'verbose', False), ('shuffle', 'parameter', 'shuffle', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'Normalizer': (('norm', 'parameter', 'norm', False), ('copy', 'parameter', 'copy', False)),
    'OneHotEncoder': (('categories', 'parameter', 'categories', False), ('drop', 'parameter', 'drop', False), ('sparse_output', 'parameter', 'sparse_output', False), ('dtype', 'parameter', 'dtype', False), ('handle_unknown', 'parameter', 'handle_unknown', False), ('min_frequency', 'parameter', 'min_frequency', False), ('max_categories', 'parameter', 'max_categories', False), ('feature_name_combiner', 'parameter', 'feature_name_combiner', False)),
    'OrdinalEncoder': (('categories', 'parameter', 'categories', False), ('dtype', 'parameter', 'dtype', False), ('handle_unknown', 'parameter', 'handle_unknown', False), ('unknown_value', 'parameter', 'unknown_value', False), ('encoded_missing_value', 'parameter', 'encoded_missing_value', False), ('min_frequency', 'parameter', 'min_frequency', False), ('max_categories', 'parameter', 'max_categories', False)),
    'PLSCanonical': (('algorithm', 'parameter', 'algorithm', False), ('n_components', 'parameter', 'n_components', False), ('scale', 'parameter', 'scale', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('copy', 'parameter', 'copy', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'PLSRegression': (('n_components', 'parameter', 'n_components', False), ('scale', 'parameter', 'scale', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('copy', 'parameter', 'copy', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'PermutationTestResult': (('statistic', 'parameter', 'statistic', False), ('null_distribution', 'parameter', 'null_distribution', False), ('pvalue', 'parameter', 'pvalue', False), ('count_less', 'parameter', 'count_less', False), ('count_greater', 'parameter', 'count_greater', False)),
    'PolynomialFeatures': (('degree', 'parameter', 'degree', False), ('interaction_only', 'parameter', 'interaction_only', False), ('include_bias', 'parameter', 'include_bias', False), ('order', 'parameter', 'order', False)),
    'PowerTransformer': (('method', 'parameter', 'method', False), ('standardize', 'parameter', 'standardize', False), ('copy', 'parameter', 'copy', False)),
    'QuadraticDiscriminantAnalysis': (('solver', 'parameter', 'solver', False), ('shrinkage', 'parameter', 'shrinkage', False), ('priors', 'parameter', 'priors', False), ('reg_param', 'parameter', 'reg_param', False), ('store_covariance', 'parameter', 'store_covariance', False), ('tol', 'parameter', 'tol', False), ('covariance_estimator', 'parameter', 'covariance_estimator', False)),
    'QuantileTransformer': (('n_quantiles', 'parameter', 'n_quantiles', False), ('output_distribution', 'parameter', 'output_distribution', False), ('ignore_implicit_zeros', 'parameter', 'ignore_implicit_zeros', False), ('subsample', 'parameter', 'subsample', False), ('random_state', 'parameter', 'random_state', False), ('copy', 'parameter', 'copy', False)),
    'RFE': (('estimator', 'parameter', 'estimator', False), ('n_features_to_select', 'parameter', 'n_features_to_select', False), ('step', 'parameter', 'step', False), ('verbose', 'parameter', 'verbose', False), ('importance_getter', 'parameter', 'importance_getter', False)),
    'SelectKBest': (('score_func', 'parameter', 'score_func', False), ('k', 'parameter', 'k', False)),
    'ShuffleSplit': (('n_splits', 'parameter', 'n_splits', False), ('test_size', 'parameter', 'test_size', False), ('train_size', 'parameter', 'train_size', False), ('random_state', 'parameter', 'random_state', False)),
    'SimpleImputer': (('missing_values', 'parameter', 'missing_values', False), ('strategy', 'parameter', 'strategy', False), ('fill_value', 'parameter', 'fill_value', False), ('copy', 'parameter', 'copy', False), ('add_indicator', 'parameter', 'add_indicator', False), ('keep_empty_features', 'parameter', 'keep_empty_features', False)),
    'SparseCoder': (('dictionary', 'parameter', 'dictionary', False), ('transform_algorithm', 'parameter', 'transform_algorithm', False), ('transform_n_nonzero_coefs', 'parameter', 'transform_n_nonzero_coefs', False), ('transform_alpha', 'parameter', 'transform_alpha', False), ('split_sign', 'parameter', 'split_sign', False), ('n_jobs', 'parameter', 'n_jobs', False), ('positive_code', 'parameter', 'positive_code', False), ('transform_max_iter', 'parameter', 'transform_max_iter', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SparsePCA': (('n_components', 'parameter', 'n_components', False), ('alpha', 'parameter', 'alpha', False), ('ridge_alpha', 'parameter', 'ridge_alpha', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('method', 'parameter', 'method', False), ('n_jobs', 'parameter', 'n_jobs', False), ('U_init', 'parameter', 'U_init', False), ('V_init', 'parameter', 'V_init', False), ('verbose', 'parameter', 'verbose', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SparseRandomProjection': (('n_components', 'parameter', 'n_components', False), ('density', 'parameter', 'density', False), ('eps', 'parameter', 'eps', False), ('dense_output', 'parameter', 'dense_output', False), ('compute_inverse_components', 'parameter', 'compute_inverse_components', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SplineTransformer': (('n_knots', 'parameter', 'n_knots', False), ('degree', 'parameter', 'degree', False), ('knots', 'parameter', 'knots', False), ('extrapolation', 'parameter', 'extrapolation', False), ('include_bias', 'parameter', 'include_bias', False), ('order', 'parameter', 'order', False), ('handle_missing', 'parameter', 'handle_missing', False), ('sparse_output', 'parameter', 'sparse_output', False)),
    'StratifiedShuffleSplit': (('n_splits', 'parameter', 'n_splits', False), ('test_size', 'parameter', 'test_size', False), ('train_size', 'parameter', 'train_size', False), ('random_state', 'parameter', 'random_state', False)),
    'TargetEncoder': (('categories', 'parameter', 'categories', False), ('target_type', 'parameter', 'target_type', False), ('smooth', 'parameter', 'smooth', False), ('cv', 'parameter', 'cv', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False)),
    'VarianceThreshold': (('threshold', 'parameter', 'threshold', False),),
}


def _initializer(owner):
    def __init__(self, *args, **kwargs):
        return _ENTRY(self, args, kwargs, _CONTEXT, owner)
    return __init__


def install():
    # glue: install before every class-layout snapshot; signature/default
    # metadata is copied at import and the original body is kept only as the
    # test oracle.
    import inspect
    from ._native_public_attributes import _resolved
    for cls in _CLASSES:  # glue: exact source-derived field-only constructor contracts
        original = _resolved(cls, '__init__', _CONTEXT['missing'])
        if hasattr(original, '__wrapped__'):
            raise TypeError('declared native constructor requires an unwrapped source constructor')
        signature = inspect.signature(original)
        parameters = tuple(signature.parameters.values())[1:]
        if any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in parameters):  # glue: import-time signature admission
            raise TypeError('declared native constructor requires an ordinary declared signature')
        names = tuple(p.name for p in parameters)  # glue: exact parameter metadata, borrowed defaults
        events = _EVENTS[cls.__name__]
        if any(kind == 'parameter' and value not in names for _, kind, value, _ in events):  # glue: refuse stale constructor schemas
            raise TypeError('declared constructor schema differs from its constructor signature')
        _CONTEXT['plans'][cls] = dict(names=names, events=events, signature=signature,
            defaults={p.name: p.default for p in parameters if p.default is not p.empty},  # glue: exact default reference identities
            positional=tuple(p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD),  # glue: native argument schema
            kwonly=tuple(p.name for p in parameters if p.kind is p.KEYWORD_ONLY),  # glue: native argument schema
            required_positional=tuple(p.name for p in parameters  # glue: native argument schema
                                      if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is p.empty),
            required_kwonly=tuple(p.name for p in parameters  # glue: native argument schema
                                  if p.kind is p.KEYWORD_ONLY and p.default is p.empty),
            rule_fields=(), label=getattr(original.__code__, 'co_qualname', original.__qualname__))
        _CONTEXT['originals'][cls] = original
    for cls in _CLASSES:  # glue: one direct native-call constructor per class
        original = _CONTEXT['originals'][cls]
        function = _initializer(cls)
        function.__name__ = '__init__'
        function.__qualname__ = cls.__qualname__ + '.__init__'
        function.__module__ = cls.__module__
        function.__doc__ = original.__doc__
        function.__signature__ = _CONTEXT['plans'][cls]['signature']
        cls.__init__ = function


def connect(declared):
    # glue: final storage/descriptor identities after the native attribute
    # slots and the declared setter context exist.
    from ._native_public_attributes import _resolved, _CONTEXT as attributes
    for cls in _CLASSES:  # glue: fixed post-install layout snapshots
        plan = _CONTEXT['plans'][cls]
        mro = type.__dict__['__mro__'].__get__(cls)
        fields = tuple(dict.fromkeys(event[0] for event in plan['events']))  # glue: constructor destinations in first-insertion order
        if _resolved(cls, '__setattr__', _CONTEXT['missing']) is not object.__setattr__:
            raise TypeError('declared native constructor requires ordinary instance assignment')
        plan.update(field_names=fields,
            field_hooks=tuple((base, name, vars(base).get(name)) for base in mro for name in fields),  # glue: parameter/constant field descriptors
            constructor_hooks=tuple((name, _resolved(cls, name, _CONTEXT['missing']))  # glue: final lookup/assignment slots
                                    for name in _CONSTRUCTOR_HOOKS))  # glue: three fixed slot names
    hook_types = [type(vars(object)['__getattribute__']), type(vars(object)['__setattr__'])]
    if 'descriptor' in attributes:
        hook_types.append(type(attributes['descriptor']))
    _CONTEXT.update(declared=declared, constructor_hook_types=tuple(hook_types), ready=True)
