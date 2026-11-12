# SPDX-License-Identifier: Apache-2.0
"""Import-time schemas for full-reset field-only parameter setters."""
from types import BuiltinFunctionType
from .tokenizer import _binding

_ENTRY = _binding()._module.full_reset_params_native_call
if type(_ENTRY) is not BuiltinFunctionType:
    raise TypeError('full-reset parameter setters require a compiled native entry')
# The same event schemas are the whole original constructors: the public
# __init__ of every class below is one call to this entry
# (bindings/full_reset_constructor_native.mojo).
_INIT_ENTRY = _binding()._module.full_reset_constructor_native_call
if type(_INIT_ENTRY) is not BuiltinFunctionType:
    raise TypeError('full-reset constructors require a compiled native entry')
# The same context object serves the generic ParamsMixin.set_params slot
# (_mode._GENERIC_PARAMS_CONTEXT): one native owner for both setters.
from ._mode import ParamsMixin as _ParamsMixin, _GENERIC_PARAMS_CONTEXT as _CONTEXT
_CONTEXT['generic_setter'] = _ParamsMixin.set_params
# Instance lookup/assignment protocols a native constructor write must not bypass.
_CONSTRUCTOR_HOOKS = ('__getattribute__', '__getattr__', '__setattr__')
# Whole public neighbor operations have native validation, fitted-state,
# dispatch and result owners. Publish this marker only after replacing their
# constructors below: NumericModeMixin must first inject numeric_mode into the
# original signature, which the native constructor plan retains verbatim.
# Keep an explicit vetted inventory; field-only construction alone does not
# establish that the remaining methods of every _CLASSES member are native.
# Class names, resolved through _CLASS_SOURCES when their family loads.
_NATIVE_PUBLIC_CLASSES = (
    'NearestNeighbors',
    'KNeighborsClassifier',
    'KNeighborsRegressor',
    'RadiusNeighbors',
    'LocalOutlierFactor',
    'NearestCentroid',
    'IVFIndex',
    'IVFPQIndex',
    'IVFSQIndex',
    'IVFRaBitQIndex',
    'CagraIndex',
)
# Native archives below also replace the old post-load fit wrapper. Their
# sole package hook (NearestNeighbors' empty resident holder) is published by
# host_archive_native; the connected identity refuses changed/user hooks.
_NATIVE_PUBLIC_LOADERS = (
    'NearestNeighbors',
    'KNeighborsClassifier',
    'KNeighborsRegressor',
    'RadiusNeighbors',
    'IVFIndex',
)
# Class name -> the package module that defines it (lane lazy-family-import,
# 2026-11-12): a family is connected when it loads (`collect()` then
# `install()`/`connect()` from `_families`), never imported by this module.
_CLASS_SOURCES = {
    'AdaBoostClassifier': '_expansion_trees',
    'AdaBoostRegressor': '_expansion_trees',
    'AdditiveChi2Sampler': '_expansion_neighbors',
    'AffinityPropagation': '_expansion_cluster',
    'AgglomerativeClustering': '_hierarchy_impl',
    'BaggingClassifier': '_expansion_trees',
    'BaggingRegressor': '_expansion_trees',
    'BayesianGaussianMixture': '_expansion_cluster',
    'BisectingKMeans': '_expansion_cluster',
    'CagraIndex': '_expansion_ann',
    'CalibratedClassifierCV': '_expansion_trees',
    'DBSCAN': 'density',
    'ElasticNet': '_solver_impl',
    'GaussianMixture': 'mixture',
    'GaussianProcessClassifier': '_gpc_impl',
    'GaussianProcessRegressor': '_gp_impl',
    'HDBSCAN': 'hdbscan',
    'IVFIndex': '_ivf_impl',
    'IVFPQIndex': '_expansion_ann',
    'IVFRaBitQIndex': '_expansion_ann',
    'IVFSQIndex': '_expansion_ann',
    'IsolationForest': '_iforest_impl',
    'IsotonicRegression': '_expansion_linear',
    'KMeans': 'cluster',
    'KNNImputer': '_expansion_neighbors',
    'KNeighborsClassifier': 'neighbors',
    'KNeighborsRegressor': 'neighbors',
    'KernelDensity': 'density',
    'KernelPCA': '_expansion_neighbors',
    'KernelRidge': 'kernel_methods',
    'LabelPropagation': '_expansion_neighbors',
    'LabelSpreading': '_expansion_neighbors',
    'Lasso': '_solver_impl',
    'LinearSVC': 'svm',
    'LinearSVR': 'svm',
    'LocalOutlierFactor': '_expansion_neighbors',
    'Louvain': '_expansion_neighbors',
    'MeanShift': '_expansion_cluster',
    'MiniBatchKMeans': '_expansion_cluster',
    'MultiOutputClassifier': '_expansion_trees',
    'MultiOutputRegressor': '_expansion_trees',
    'NearestCentroid': '_expansion_neighbors',
    'NearestNeighbors': 'neighbors',
    'Nystroem': 'kernel_methods',
    'OPTICS': '_expansion_cluster',
    'OneClassSVM': '_expansion_neighbors',
    'OneVsRestClassifier': '_expansion_trees',
    'PCA': 'decomposition',
    'PageRank': '_expansion_neighbors',
    'PolynomialCountSketch': '_expansion_neighbors',
    'RBFSampler': 'kernel_methods',
    'RadiusNeighbors': 'neighbors',
    'RandomTreesEmbedding': '_expansion_trees',
    'SGDOneClassSVM': '_expansion_linear',
    'SVC': '_svm_impl',
    'SVGP': '_expansion_neighbors',
    'SVR': '_svm_impl',
    'SkewedChi2Sampler': '_expansion_neighbors',
    'SpectralClustering': '_spectral_impl',
    'SpectralEmbedding': '_spectral_impl',
    'StackingClassifier': '_expansion_trees',
    'StackingRegressor': '_expansion_trees',
    'TSNE': '_expansion_ann',
    'TruncatedSVD': 'decomposition',
    'UMAP': '_umap_impl',
    'VotingClassifier': '_expansion_trees',
    'VotingRegressor': '_expansion_trees',
}
# The loaded classes of _CLASS_SOURCES, in its order (`collect()`).
_CLASSES = ()
# Modules that need NumPy at import stay lazy: `import mojolearn` never imports
# NumPy. Each one calls connect_module(globals()) as its last statement, which
# connects these classes with the same schemas, order and native context.
_DEFERRED_MODULES = {
    '_x_sequence_croston': ('CrostonClassic', 'CrostonOptimized', 'CrostonSBA'),
    '_x_sequence_rnn': ('GRUClassifier', 'GRURegressor', 'LSTMClassifier', 'LSTMRegressor',
                        'RNNClassifier', 'RNNRegressor'),
    '_x_sequence_mlp': ('MLPClassifier', 'MLPRegressor'),
}
# Constructor write events: field, parameter/constant, borrowed value/name,
# NumericModeMixin post-constructor rule. Repeated events preserve insertion
# order; fixed nonparameter fields are part of the original constructor state.
# A _mojolearn_native_public class has no NumericModeMixin wrapper: its own
# constructor body writes numeric_mode unconditionally, so that event carries
# no rule (False), as tools/native_full_reset_constructor_schemas.json audits.
_EVENTS = {
    'AdaBoostClassifier': (('estimator', 'parameter', 'estimator', False), ('n_estimators', 'parameter', 'n_estimators', False), ('learning_rate', 'parameter', 'learning_rate', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'constant', None, True), ('algorithm', 'parameter', 'algorithm', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'AdaBoostRegressor': (('estimator', 'parameter', 'estimator', False), ('n_estimators', 'parameter', 'n_estimators', False), ('learning_rate', 'parameter', 'learning_rate', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'constant', None, True), ('loss', 'parameter', 'loss', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'AdditiveChi2Sampler': (('sample_steps', 'parameter', 'sample_steps', False), ('sample_interval', 'parameter', 'sample_interval', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'AffinityPropagation': (('damping', 'parameter', 'damping', False), ('max_iter', 'parameter', 'max_iter', False), ('convergence_iter', 'parameter', 'convergence_iter', False), ('copy', 'parameter', 'copy', False), ('preference', 'parameter', 'preference', False), ('affinity', 'parameter', 'affinity', False), ('verbose', 'parameter', 'verbose', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'AgglomerativeClustering': (('n_clusters', 'parameter', 'n_clusters', False), ('metric', 'parameter', 'metric', False), ('connectivity', 'parameter', 'connectivity', False), ('linkage', 'parameter', 'linkage', False), ('c', 'parameter', 'c', False), ('memory', 'parameter', 'memory', False), ('compute_full_tree', 'parameter', 'compute_full_tree', False), ('distance_threshold', 'parameter', 'distance_threshold', False), ('compute_distances', 'parameter', 'compute_distances', False), ('prediction_data', 'parameter', 'prediction_data', False)),
    'BaggingClassifier': (('estimator', 'parameter', 'estimator', False), ('n_estimators', 'parameter', 'n_estimators', False), ('max_samples', 'parameter', 'max_samples', False), ('max_features', 'parameter', 'max_features', False), ('bootstrap', 'parameter', 'bootstrap', False), ('bootstrap_features', 'parameter', 'bootstrap_features', False), ('oob_score', 'parameter', 'oob_score', False), ('warm_start', 'parameter', 'warm_start', False), ('n_jobs', 'parameter', 'n_jobs', False), ('random_state', 'parameter', 'random_state', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'constant', None, True), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'BaggingRegressor': (('estimator', 'parameter', 'estimator', False), ('n_estimators', 'parameter', 'n_estimators', False), ('max_samples', 'parameter', 'max_samples', False), ('max_features', 'parameter', 'max_features', False), ('bootstrap', 'parameter', 'bootstrap', False), ('bootstrap_features', 'parameter', 'bootstrap_features', False), ('oob_score', 'parameter', 'oob_score', False), ('warm_start', 'parameter', 'warm_start', False), ('n_jobs', 'parameter', 'n_jobs', False), ('random_state', 'parameter', 'random_state', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'constant', None, True), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'BayesianGaussianMixture': (('n_components', 'parameter', 'n_components', False), ('covariance_type', 'parameter', 'covariance_type', False), ('tol', 'parameter', 'tol', False), ('reg_covar', 'parameter', 'reg_covar', False), ('max_iter', 'parameter', 'max_iter', False), ('n_init', 'parameter', 'n_init', False), ('init_params', 'parameter', 'init_params', False), ('weight_concentration_prior_type', 'parameter', 'weight_concentration_prior_type', False), ('weight_concentration_prior', 'parameter', 'weight_concentration_prior', False), ('mean_precision_prior', 'parameter', 'mean_precision_prior', False), ('mean_prior', 'parameter', 'mean_prior', False), ('degrees_of_freedom_prior', 'parameter', 'degrees_of_freedom_prior', False), ('covariance_prior', 'parameter', 'covariance_prior', False), ('random_state', 'parameter', 'random_state', False), ('warm_start', 'parameter', 'warm_start', False), ('verbose', 'parameter', 'verbose', False), ('verbose_interval', 'parameter', 'verbose_interval', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'BisectingKMeans': (('n_clusters', 'parameter', 'n_clusters', False), ('init', 'parameter', 'init', False), ('n_init', 'parameter', 'n_init', False), ('random_state', 'parameter', 'random_state', False), ('max_iter', 'parameter', 'max_iter', False), ('verbose', 'parameter', 'verbose', False), ('tol', 'parameter', 'tol', False), ('copy_x', 'parameter', 'copy_x', False), ('algorithm', 'parameter', 'algorithm', False), ('bisecting_strategy', 'parameter', 'bisecting_strategy', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'CagraIndex': (('graph_degree', 'parameter', 'graph_degree', False), ('intermediate_graph_degree', 'parameter', 'intermediate_graph_degree', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('itopk_size', 'parameter', 'itopk_size', False), ('search_width', 'parameter', 'search_width', False), ('max_iterations', 'parameter', 'max_iterations', False), ('n_seeds', 'parameter', 'n_seeds', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'CalibratedClassifierCV': (('estimator', 'parameter', 'estimator', False), ('method', 'parameter', 'method', False), ('cv', 'parameter', 'cv', False), ('n_jobs', 'parameter', 'n_jobs', False), ('ensemble', 'parameter', 'ensemble', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'CrostonClassic': (('alias', 'parameter', 'alias', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'CrostonOptimized': (('alias', 'parameter', 'alias', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'CrostonSBA': (('alias', 'parameter', 'alias', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'DBSCAN': (('eps', 'parameter', 'eps', False), ('min_samples', 'parameter', 'min_samples', False), ('metric', 'parameter', 'metric', False), ('algorithm', 'parameter', 'algorithm', False), ('max_mbytes_per_batch', 'parameter', 'max_mbytes_per_batch', False), ('max_iterations', 'parameter', 'max_iterations', False), ('prediction_data', 'parameter', 'prediction_data', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'ElasticNet': (('alpha', 'parameter', 'alpha', False), ('l1_ratio', 'parameter', 'l1_ratio', False), ('fit_intercept', 'parameter', 'fit_intercept', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('solver', 'parameter', 'solver', False), ('selection', 'parameter', 'selection', False), ('precompute', 'parameter', 'precompute', False), ('copy_X', 'parameter', 'copy_X', False), ('warm_start', 'parameter', 'warm_start', False), ('positive', 'parameter', 'positive', False), ('random_state', 'parameter', 'random_state', False)),
    'GRUClassifier': (('hidden_size', 'parameter', 'hidden_size', False), ('num_layers', 'parameter', 'num_layers', False), ('optimizer', 'parameter', 'optimizer', False), ('learning_rate', 'parameter', 'learning_rate', False), ('batch_size', 'parameter', 'batch_size', False), ('max_epochs', 'parameter', 'max_epochs', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('optimizer_options', 'parameter', 'optimizer_options', False), ('nonlinearity', 'parameter', 'nonlinearity', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('predict_chunk', 'parameter', 'predict_chunk', False), ('lr_schedule', 'parameter', 'lr_schedule', False)),
    'GRURegressor': (('hidden_size', 'parameter', 'hidden_size', False), ('num_layers', 'parameter', 'num_layers', False), ('optimizer', 'parameter', 'optimizer', False), ('learning_rate', 'parameter', 'learning_rate', False), ('batch_size', 'parameter', 'batch_size', False), ('max_epochs', 'parameter', 'max_epochs', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('optimizer_options', 'parameter', 'optimizer_options', False), ('nonlinearity', 'parameter', 'nonlinearity', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('predict_chunk', 'parameter', 'predict_chunk', False), ('lr_schedule', 'parameter', 'lr_schedule', False)),
    'GaussianMixture': (('n_components', 'parameter', 'n_components', False), ('covariance_type', 'parameter', 'covariance_type', False), ('tol', 'parameter', 'tol', False), ('reg_covar', 'parameter', 'reg_covar', False), ('max_iter', 'parameter', 'max_iter', False), ('init_params', 'parameter', 'init_params', False), ('random_state', 'parameter', 'random_state', False), ('n_init', 'parameter', 'n_init', False), ('warm_start', 'parameter', 'warm_start', False), ('means_init', 'parameter', 'means_init', False), ('weights_init', 'parameter', 'weights_init', False), ('precisions_init', 'parameter', 'precisions_init', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'GaussianProcessClassifier': (('kernel', 'parameter', 'kernel', False), ('optimizer', 'parameter', 'optimizer', False), ('n_restarts_optimizer', 'parameter', 'n_restarts_optimizer', False), ('max_iter_predict', 'parameter', 'max_iter_predict', False), ('warm_start', 'parameter', 'warm_start', False), ('copy_X_train', 'parameter', 'copy_X_train', False), ('random_state', 'parameter', 'random_state', False), ('multi_class', 'parameter', 'multi_class', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'GaussianProcessRegressor': (('kernel', 'parameter', 'kernel', False), ('alpha', 'parameter', 'alpha', False), ('optimizer', 'parameter', 'optimizer', False), ('n_restarts_optimizer', 'parameter', 'n_restarts_optimizer', False), ('normalize_y', 'parameter', 'normalize_y', False), ('copy_X_train', 'parameter', 'copy_X_train', False), ('n_targets', 'parameter', 'n_targets', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'HDBSCAN': (('min_cluster_size', 'parameter', 'min_cluster_size', False), ('min_samples', 'parameter', 'min_samples', False), ('cluster_selection_epsilon', 'parameter', 'cluster_selection_epsilon', False), ('max_cluster_size', 'parameter', 'max_cluster_size', False), ('metric', 'parameter', 'metric', False), ('alpha', 'parameter', 'alpha', False), ('cluster_selection_method', 'parameter', 'cluster_selection_method', False), ('allow_single_cluster', 'parameter', 'allow_single_cluster', False), ('prediction_data', 'parameter', 'prediction_data', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'IVFIndex': (('n_lists', 'parameter', 'n_lists', False), ('n_probes', 'parameter', 'n_probes', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('kmeans_n_iters', 'parameter', 'kmeans_n_iters', False), ('metric', 'parameter', 'metric', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'IVFPQIndex': (('n_lists', 'parameter', 'n_lists', False), ('n_probes', 'parameter', 'n_probes', False), ('pq_dim', 'parameter', 'pq_dim', False), ('pq_bits', 'parameter', 'pq_bits', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('kmeans_n_iters', 'parameter', 'kmeans_n_iters', False), ('pq_kmeans_n_iters', 'parameter', 'pq_kmeans_n_iters', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'IVFRaBitQIndex': (('n_lists', 'parameter', 'n_lists', False), ('n_probes', 'parameter', 'n_probes', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('kmeans_n_iters', 'parameter', 'kmeans_n_iters', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'IVFSQIndex': (('n_lists', 'parameter', 'n_lists', False), ('n_probes', 'parameter', 'n_probes', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('kmeans_n_iters', 'parameter', 'kmeans_n_iters', False), ('random_state', 'parameter', 'random_state', False), ('max_train_points_per_cluster', 'parameter', 'max_train_points_per_cluster', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'IsolationForest': (('n_estimators', 'parameter', 'n_estimators', False), ('max_samples', 'parameter', 'max_samples', False), ('max_depth', 'parameter', 'max_depth', False), ('max_features', 'parameter', 'max_features', False), ('bootstrap', 'parameter', 'bootstrap', False), ('contamination', 'parameter', 'contamination', False), ('random_state', 'parameter', 'random_state', False), ('warm_start', 'parameter', 'warm_start', False), ('n_jobs', 'parameter', 'n_jobs', False), ('verbose', 'parameter', 'verbose', False), ('output_type', 'parameter', 'output_type', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'IsotonicRegression': (('y_min', 'parameter', 'y_min', False), ('y_max', 'parameter', 'y_max', False), ('increasing', 'parameter', 'increasing', False), ('out_of_bounds', 'parameter', 'out_of_bounds', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'KMeans': (('n_clusters', 'parameter', 'n_clusters', False), ('init', 'parameter', 'init', False), ('n_init', 'parameter', 'n_init', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('random_state', 'parameter', 'random_state', False), ('init_centroids', 'parameter', 'init_centroids', False), ('metric', 'parameter', 'metric', False), ('oversampling_factor', 'parameter', 'oversampling_factor', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'KNNImputer': (('missing_values', 'parameter', 'missing_values', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('weights', 'parameter', 'weights', False), ('metric', 'parameter', 'metric', False), ('copy', 'parameter', 'copy', False), ('add_indicator', 'parameter', 'add_indicator', False), ('keep_empty_features', 'parameter', 'keep_empty_features', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'KNeighborsClassifier': (('n_neighbors', 'parameter', 'n_neighbors', False), ('query_tile', 'parameter', 'query_tile', False), ('metric', 'parameter', 'metric', False), ('algorithm', 'parameter', 'algorithm', False), ('p', 'parameter', 'p', False), ('_index', 'constant', None, False), ('numeric_mode', 'constant', None, True), ('weights', 'parameter', 'weights', False), ('_y_cols', 'constant', None, False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'KNeighborsRegressor': (('n_neighbors', 'parameter', 'n_neighbors', False), ('query_tile', 'parameter', 'query_tile', False), ('metric', 'parameter', 'metric', False), ('algorithm', 'parameter', 'algorithm', False), ('p', 'parameter', 'p', False), ('_index', 'constant', None, False), ('numeric_mode', 'constant', None, True), ('weights', 'parameter', 'weights', False), ('_y_cols', 'constant', None, False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'KernelDensity': (('bandwidth', 'parameter', 'bandwidth', False), ('algorithm', 'parameter', 'algorithm', False), ('kernel', 'parameter', 'kernel', False), ('metric', 'parameter', 'metric', False), ('atol', 'parameter', 'atol', False), ('rtol', 'parameter', 'rtol', False), ('breadth_first', 'parameter', 'breadth_first', False), ('leaf_size', 'parameter', 'leaf_size', False), ('metric_params', 'parameter', 'metric_params', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'KernelPCA': (('n_components', 'parameter', 'n_components', False), ('kernel', 'parameter', 'kernel', False), ('gamma', 'parameter', 'gamma', False), ('degree', 'parameter', 'degree', False), ('coef0', 'parameter', 'coef0', False), ('kernel_params', 'parameter', 'kernel_params', False), ('alpha', 'parameter', 'alpha', False), ('fit_inverse_transform', 'parameter', 'fit_inverse_transform', False), ('eigen_solver', 'parameter', 'eigen_solver', False), ('tol', 'parameter', 'tol', False), ('max_iter', 'parameter', 'max_iter', False), ('iterated_power', 'parameter', 'iterated_power', False), ('remove_zero_eig', 'parameter', 'remove_zero_eig', False), ('random_state', 'parameter', 'random_state', False), ('copy_X', 'parameter', 'copy_X', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'KernelRidge': (('alpha', 'parameter', 'alpha', False), ('kernel', 'parameter', 'kernel', False), ('gamma', 'parameter', 'gamma', False), ('degree', 'parameter', 'degree', False), ('coef0', 'parameter', 'coef0', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'LSTMClassifier': (('hidden_size', 'parameter', 'hidden_size', False), ('num_layers', 'parameter', 'num_layers', False), ('optimizer', 'parameter', 'optimizer', False), ('learning_rate', 'parameter', 'learning_rate', False), ('batch_size', 'parameter', 'batch_size', False), ('max_epochs', 'parameter', 'max_epochs', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('optimizer_options', 'parameter', 'optimizer_options', False), ('nonlinearity', 'parameter', 'nonlinearity', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('predict_chunk', 'parameter', 'predict_chunk', False), ('lr_schedule', 'parameter', 'lr_schedule', False)),
    'LSTMRegressor': (('hidden_size', 'parameter', 'hidden_size', False), ('num_layers', 'parameter', 'num_layers', False), ('optimizer', 'parameter', 'optimizer', False), ('learning_rate', 'parameter', 'learning_rate', False), ('batch_size', 'parameter', 'batch_size', False), ('max_epochs', 'parameter', 'max_epochs', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('optimizer_options', 'parameter', 'optimizer_options', False), ('nonlinearity', 'parameter', 'nonlinearity', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('predict_chunk', 'parameter', 'predict_chunk', False), ('lr_schedule', 'parameter', 'lr_schedule', False)),
    'LabelPropagation': (('kernel', 'parameter', 'kernel', False), ('gamma', 'parameter', 'gamma', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'LabelSpreading': (('kernel', 'parameter', 'kernel', False), ('gamma', 'parameter', 'gamma', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('alpha', 'parameter', 'alpha', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'Lasso': (('alpha', 'parameter', 'alpha', False), ('l1_ratio', 'constant', 1.0, False), ('fit_intercept', 'parameter', 'fit_intercept', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('solver', 'parameter', 'solver', False), ('selection', 'parameter', 'selection', False), ('precompute', 'parameter', 'precompute', False), ('copy_X', 'parameter', 'copy_X', False), ('warm_start', 'parameter', 'warm_start', False), ('positive', 'parameter', 'positive', False), ('random_state', 'parameter', 'random_state', False)),
    'LinearSVC': (('penalty', 'parameter', 'penalty', False), ('loss', 'parameter', 'loss', False), ('C', 'parameter', 'C', False), ('fit_intercept', 'parameter', 'fit_intercept', False), ('penalized_intercept', 'parameter', 'penalized_intercept', False), ('tol', 'parameter', 'tol', False), ('max_iter', 'parameter', 'max_iter', False), ('linesearch_max_iter', 'parameter', 'linesearch_max_iter', False), ('lbfgs_memory', 'parameter', 'lbfgs_memory', False), ('class_weight', 'parameter', 'class_weight', False), ('multi_class', 'parameter', 'multi_class', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'LinearSVR': (('penalty', 'parameter', 'penalty', False), ('loss', 'parameter', 'loss', False), ('C', 'parameter', 'C', False), ('fit_intercept', 'parameter', 'fit_intercept', False), ('penalized_intercept', 'parameter', 'penalized_intercept', False), ('tol', 'parameter', 'tol', False), ('max_iter', 'parameter', 'max_iter', False), ('linesearch_max_iter', 'parameter', 'linesearch_max_iter', False), ('lbfgs_memory', 'parameter', 'lbfgs_memory', False), ('epsilon', 'parameter', 'epsilon', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'LocalOutlierFactor': (('n_neighbors', 'parameter', 'n_neighbors', False), ('algorithm', 'parameter', 'algorithm', False), ('leaf_size', 'parameter', 'leaf_size', False), ('metric', 'parameter', 'metric', False), ('p', 'parameter', 'p', False), ('metric_params', 'parameter', 'metric_params', False), ('contamination', 'parameter', 'contamination', False), ('novelty', 'parameter', 'novelty', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'Louvain': (('resolution', 'parameter', 'resolution', False), ('threshold', 'parameter', 'threshold', False), ('max_level', 'parameter', 'max_level', False), ('seed', 'parameter', 'seed', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MLPClassifier': (('hidden_layer_sizes', 'parameter', 'hidden_layer_sizes', False), ('activation', 'parameter', 'activation', False), ('solver', 'parameter', 'solver', False), ('alpha', 'parameter', 'alpha', False), ('batch_size', 'parameter', 'batch_size', False), ('learning_rate', 'parameter', 'learning_rate', False), ('learning_rate_init', 'parameter', 'learning_rate_init', False), ('power_t', 'parameter', 'power_t', False), ('max_iter', 'parameter', 'max_iter', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('tol', 'parameter', 'tol', False), ('verbose', 'parameter', 'verbose', False), ('warm_start', 'parameter', 'warm_start', False), ('momentum', 'parameter', 'momentum', False), ('nesterovs_momentum', 'parameter', 'nesterovs_momentum', False), ('early_stopping', 'parameter', 'early_stopping', False), ('validation_fraction', 'parameter', 'validation_fraction', False), ('beta_1', 'parameter', 'beta_1', False), ('beta_2', 'parameter', 'beta_2', False), ('epsilon', 'parameter', 'epsilon', False), ('n_iter_no_change', 'parameter', 'n_iter_no_change', False), ('max_fun', 'parameter', 'max_fun', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MLPRegressor': (('hidden_layer_sizes', 'parameter', 'hidden_layer_sizes', False), ('activation', 'parameter', 'activation', False), ('solver', 'parameter', 'solver', False), ('alpha', 'parameter', 'alpha', False), ('batch_size', 'parameter', 'batch_size', False), ('learning_rate', 'parameter', 'learning_rate', False), ('learning_rate_init', 'parameter', 'learning_rate_init', False), ('power_t', 'parameter', 'power_t', False), ('max_iter', 'parameter', 'max_iter', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('tol', 'parameter', 'tol', False), ('verbose', 'parameter', 'verbose', False), ('warm_start', 'parameter', 'warm_start', False), ('momentum', 'parameter', 'momentum', False), ('nesterovs_momentum', 'parameter', 'nesterovs_momentum', False), ('early_stopping', 'parameter', 'early_stopping', False), ('validation_fraction', 'parameter', 'validation_fraction', False), ('beta_1', 'parameter', 'beta_1', False), ('beta_2', 'parameter', 'beta_2', False), ('epsilon', 'parameter', 'epsilon', False), ('n_iter_no_change', 'parameter', 'n_iter_no_change', False), ('max_fun', 'parameter', 'max_fun', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MeanShift': (('bandwidth', 'parameter', 'bandwidth', False), ('seeds', 'parameter', 'seeds', False), ('bin_seeding', 'parameter', 'bin_seeding', False), ('min_bin_freq', 'parameter', 'min_bin_freq', False), ('cluster_all', 'parameter', 'cluster_all', False), ('n_jobs', 'parameter', 'n_jobs', False), ('max_iter', 'parameter', 'max_iter', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MiniBatchKMeans': (('n_clusters', 'parameter', 'n_clusters', False), ('init', 'parameter', 'init', False), ('max_iter', 'parameter', 'max_iter', False), ('batch_size', 'parameter', 'batch_size', False), ('verbose', 'parameter', 'verbose', False), ('compute_labels', 'parameter', 'compute_labels', False), ('random_state', 'parameter', 'random_state', False), ('tol', 'parameter', 'tol', False), ('max_no_improvement', 'parameter', 'max_no_improvement', False), ('init_size', 'parameter', 'init_size', False), ('n_init', 'parameter', 'n_init', False), ('reassignment_ratio', 'parameter', 'reassignment_ratio', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'MultiOutputClassifier': (('estimator', 'parameter', 'estimator', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'MultiOutputRegressor': (('estimator', 'parameter', 'estimator', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'NearestCentroid': (('metric', 'parameter', 'metric', False), ('shrink_threshold', 'parameter', 'shrink_threshold', False), ('priors', 'parameter', 'priors', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'NearestNeighbors': (('n_neighbors', 'parameter', 'n_neighbors', False), ('query_tile', 'parameter', 'query_tile', False), ('metric', 'parameter', 'metric', False), ('algorithm', 'parameter', 'algorithm', False), ('p', 'parameter', 'p', False), ('_index', 'constant', None, False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'Nystroem': (('kernel', 'parameter', 'kernel', False), ('gamma', 'parameter', 'gamma', False), ('degree', 'parameter', 'degree', False), ('coef0', 'parameter', 'coef0', False), ('n_components', 'parameter', 'n_components', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'OPTICS': (('min_samples', 'parameter', 'min_samples', False), ('max_eps', 'parameter', 'max_eps', False), ('metric', 'parameter', 'metric', False), ('p', 'parameter', 'p', False), ('metric_params', 'parameter', 'metric_params', False), ('cluster_method', 'parameter', 'cluster_method', False), ('eps', 'parameter', 'eps', False), ('xi', 'parameter', 'xi', False), ('predecessor_correction', 'parameter', 'predecessor_correction', False), ('min_cluster_size', 'parameter', 'min_cluster_size', False), ('algorithm', 'parameter', 'algorithm', False), ('leaf_size', 'parameter', 'leaf_size', False), ('memory', 'parameter', 'memory', False), ('n_jobs', 'parameter', 'n_jobs', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'OneClassSVM': (('kernel', 'parameter', 'kernel', False), ('degree', 'parameter', 'degree', False), ('gamma', 'parameter', 'gamma', False), ('coef0', 'parameter', 'coef0', False), ('tol', 'parameter', 'tol', False), ('nu', 'parameter', 'nu', False), ('shrinking', 'parameter', 'shrinking', False), ('cache_size', 'parameter', 'cache_size', False), ('verbose', 'parameter', 'verbose', False), ('max_iter', 'parameter', 'max_iter', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'OneVsRestClassifier': (('estimator', 'parameter', 'estimator', False), ('n_jobs', 'parameter', 'n_jobs', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'PCA': (('n_components', 'parameter', 'n_components', False), ('copy', 'parameter', 'copy', False), ('tol', 'parameter', 'tol', False), ('whiten', 'parameter', 'whiten', False), ('svd_solver', 'parameter', 'svd_solver', False), ('iterated_power', 'parameter', 'iterated_power', False), ('n_oversamples', 'parameter', 'n_oversamples', False), ('power_iteration_normalizer', 'parameter', 'power_iteration_normalizer', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'PageRank': (('alpha', 'parameter', 'alpha', False), ('personalization', 'parameter', 'personalization', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('nstart', 'parameter', 'nstart', False), ('weight', 'parameter', 'weight', False), ('dangling', 'parameter', 'dangling', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'PolynomialCountSketch': (('gamma', 'parameter', 'gamma', False), ('degree', 'parameter', 'degree', False), ('coef0', 'parameter', 'coef0', False), ('n_components', 'parameter', 'n_components', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'RBFSampler': (('gamma', 'parameter', 'gamma', False), ('n_components', 'parameter', 'n_components', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'RNNClassifier': (('hidden_size', 'parameter', 'hidden_size', False), ('num_layers', 'parameter', 'num_layers', False), ('optimizer', 'parameter', 'optimizer', False), ('learning_rate', 'parameter', 'learning_rate', False), ('batch_size', 'parameter', 'batch_size', False), ('max_epochs', 'parameter', 'max_epochs', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('optimizer_options', 'parameter', 'optimizer_options', False), ('nonlinearity', 'parameter', 'nonlinearity', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('predict_chunk', 'parameter', 'predict_chunk', False), ('lr_schedule', 'parameter', 'lr_schedule', False)),
    'RNNRegressor': (('hidden_size', 'parameter', 'hidden_size', False), ('num_layers', 'parameter', 'num_layers', False), ('optimizer', 'parameter', 'optimizer', False), ('learning_rate', 'parameter', 'learning_rate', False), ('batch_size', 'parameter', 'batch_size', False), ('max_epochs', 'parameter', 'max_epochs', False), ('shuffle', 'parameter', 'shuffle', False), ('random_state', 'parameter', 'random_state', False), ('optimizer_options', 'parameter', 'optimizer_options', False), ('nonlinearity', 'parameter', 'nonlinearity', False), ('numeric_mode', 'parameter', 'numeric_mode', False), ('predict_chunk', 'parameter', 'predict_chunk', False), ('lr_schedule', 'parameter', 'lr_schedule', False)),
    'RadiusNeighbors': (('radius', 'parameter', 'radius', False), ('metric', 'parameter', 'metric', False), ('algorithm', 'parameter', 'algorithm', False), ('p', 'parameter', 'p', False), ('_index', 'constant', None, False), ('_metric_value', 'constant', None, False), ('_metric_arg', 'constant', None, False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'RandomTreesEmbedding': (('n_estimators', 'parameter', 'n_estimators', False), ('max_depth', 'parameter', 'max_depth', False), ('min_samples_split', 'parameter', 'min_samples_split', False), ('min_samples_leaf', 'parameter', 'min_samples_leaf', False), ('min_weight_fraction_leaf', 'parameter', 'min_weight_fraction_leaf', False), ('max_leaf_nodes', 'parameter', 'max_leaf_nodes', False), ('min_impurity_decrease', 'parameter', 'min_impurity_decrease', False), ('sparse_output', 'parameter', 'sparse_output', False), ('n_jobs', 'parameter', 'n_jobs', False), ('random_state', 'parameter', 'random_state', False), ('verbose', 'parameter', 'verbose', False), ('warm_start', 'parameter', 'warm_start', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SGDOneClassSVM': (('batch_size', 'parameter', 'batch_size', False), ('nu', 'parameter', 'nu', False), ('fit_intercept', 'parameter', 'fit_intercept', False), ('max_iter', 'parameter', 'max_iter', False), ('tol', 'parameter', 'tol', False), ('shuffle', 'parameter', 'shuffle', False), ('verbose', 'parameter', 'verbose', False), ('random_state', 'parameter', 'random_state', False), ('learning_rate', 'parameter', 'learning_rate', False), ('eta0', 'parameter', 'eta0', False), ('power_t', 'parameter', 'power_t', False), ('warm_start', 'parameter', 'warm_start', False), ('average', 'parameter', 'average', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SVC': (('C', 'parameter', 'C', False), ('kernel', 'parameter', 'kernel', False), ('degree', 'parameter', 'degree', False), ('gamma', 'parameter', 'gamma', False), ('coef0', 'parameter', 'coef0', False), ('tol', 'parameter', 'tol', False), ('cache_size', 'parameter', 'cache_size', False), ('max_iter', 'parameter', 'max_iter', False), ('nochange_steps', 'parameter', 'nochange_steps', False), ('verbose', 'parameter', 'verbose', False), ('output_type', 'parameter', 'output_type', False), ('random_state', 'parameter', 'random_state', False), ('class_weight', 'parameter', 'class_weight', False), ('decision_function_shape', 'parameter', 'decision_function_shape', False), ('probability', 'parameter', 'probability', False), ('break_ties', 'parameter', 'break_ties', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SVGP': (('n_inducing', 'parameter', 'n_inducing', False), ('inducing_points', 'parameter', 'inducing_points', False), ('kernel_variance', 'parameter', 'kernel_variance', False), ('lengthscale', 'parameter', 'lengthscale', False), ('noise_variance', 'parameter', 'noise_variance', False), ('jitter', 'parameter', 'jitter', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SVR': (('kernel', 'parameter', 'kernel', False), ('degree', 'parameter', 'degree', False), ('gamma', 'parameter', 'gamma', False), ('coef0', 'parameter', 'coef0', False), ('tol', 'parameter', 'tol', False), ('C', 'parameter', 'C', False), ('epsilon', 'parameter', 'epsilon', False), ('cache_size', 'parameter', 'cache_size', False), ('verbose', 'parameter', 'verbose', False), ('max_iter', 'parameter', 'max_iter', False), ('nochange_steps', 'parameter', 'nochange_steps', False), ('output_type', 'parameter', 'output_type', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SkewedChi2Sampler': (('skewedness', 'parameter', 'skewedness', False), ('n_components', 'parameter', 'n_components', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'SpectralClustering': (('n_clusters', 'parameter', 'n_clusters', False), ('n_components', 'parameter', 'n_components', False), ('random_state', 'parameter', 'random_state', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('n_init', 'parameter', 'n_init', False), ('eigen_tol', 'parameter', 'eigen_tol', False), ('affinity', 'parameter', 'affinity', False), ('gamma', 'parameter', 'gamma', False), ('assign_labels', 'parameter', 'assign_labels', False), ('prediction_data', 'parameter', 'prediction_data', False), ('eigen_solver', 'parameter', 'eigen_solver', False), ('degree', 'parameter', 'degree', False), ('coef0', 'parameter', 'coef0', False), ('kernel_params', 'parameter', 'kernel_params', False), ('n_jobs', 'parameter', 'n_jobs', False), ('verbose', 'parameter', 'verbose', False)),
    'SpectralEmbedding': (('n_components', 'parameter', 'n_components', False), ('affinity', 'parameter', 'affinity', False), ('random_state', 'parameter', 'random_state', False), ('n_neighbors', 'parameter', 'n_neighbors', False), ('gamma', 'parameter', 'gamma', False), ('eigen_solver', 'parameter', 'eigen_solver', False), ('eigen_tol', 'parameter', 'eigen_tol', False), ('n_jobs', 'parameter', 'n_jobs', False), ('verbose', 'parameter', 'verbose', False)),
    'StackingClassifier': (('estimators', 'parameter', 'estimators', False), ('final_estimator', 'parameter', 'final_estimator', False), ('cv', 'parameter', 'cv', False), ('stack_method', 'parameter', 'stack_method', False), ('n_jobs', 'parameter', 'n_jobs', False), ('passthrough', 'parameter', 'passthrough', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'constant', None, True), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'StackingRegressor': (('estimators', 'parameter', 'estimators', False), ('final_estimator', 'parameter', 'final_estimator', False), ('cv', 'parameter', 'cv', False), ('stack_method', 'constant', 'auto', False), ('n_jobs', 'parameter', 'n_jobs', False), ('passthrough', 'parameter', 'passthrough', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'constant', None, True), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'TSNE': (('n_components', 'parameter', 'n_components', False), ('perplexity', 'parameter', 'perplexity', False), ('early_exaggeration', 'parameter', 'early_exaggeration', False), ('learning_rate', 'parameter', 'learning_rate', False), ('post_exaggeration_learning_rate_factor', 'parameter', 'post_exaggeration_learning_rate_factor', False), ('max_iter', 'parameter', 'max_iter', False), ('init', 'parameter', 'init', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'TruncatedSVD': (('n_components', 'parameter', 'n_components', False), ('tol', 'parameter', 'tol', False), ('algorithm', 'parameter', 'algorithm', False), ('n_iter', 'parameter', 'n_iter', False), ('n_oversamples', 'parameter', 'n_oversamples', False), ('power_iteration_normalizer', 'parameter', 'power_iteration_normalizer', False), ('random_state', 'parameter', 'random_state', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'UMAP': (('n_neighbors', 'parameter', 'n_neighbors', False), ('n_components', 'parameter', 'n_components', False), ('min_dist', 'parameter', 'min_dist', False), ('spread', 'parameter', 'spread', False), ('n_epochs', 'parameter', 'n_epochs', False), ('random_state', 'parameter', 'random_state', False), ('set_op_mix_ratio', 'parameter', 'set_op_mix_ratio', False), ('local_connectivity', 'parameter', 'local_connectivity', False), ('metric', 'parameter', 'metric', False), ('init', 'parameter', 'init', False), ('learning_rate', 'parameter', 'learning_rate', False), ('repulsion_strength', 'parameter', 'repulsion_strength', False), ('negative_sample_rate', 'parameter', 'negative_sample_rate', False), ('a', 'parameter', 'a', False), ('b', 'parameter', 'b', False), ('metric_kwds', 'parameter', 'metric_kwds', False), ('target_n_neighbors', 'parameter', 'target_n_neighbors', False), ('target_metric', 'parameter', 'target_metric', False), ('target_weight', 'parameter', 'target_weight', False), ('densmap', 'parameter', 'densmap', False), ('output_metric', 'parameter', 'output_metric', False), ('numeric_mode', 'parameter', 'numeric_mode', False)),
    'VotingClassifier': (('estimators', 'parameter', 'estimators', False), ('voting', 'parameter', 'voting', False), ('weights', 'parameter', 'weights', False), ('n_jobs', 'parameter', 'n_jobs', False), ('flatten_transform', 'parameter', 'flatten_transform', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
    'VotingRegressor': (('estimators', 'parameter', 'estimators', False), ('weights', 'parameter', 'weights', False), ('n_jobs', 'parameter', 'n_jobs', False), ('verbose', 'parameter', 'verbose', False), ('numeric_mode', 'parameter', 'numeric_mode', True)),
}


def _initializer(owner):
    def __init__(self, *args, **kwargs):
        return _INIT_ENTRY(self, args, kwargs, _CONTEXT, owner)
    return __init__


class NativeFullResetParams:
    def set_params(self, **params):
        """Set constructor parameters and clear fitted state.

        Parameters
        ----------
        **params : keyword arguments
            Constructor settings; use ``name__parameter`` for nested values.

        Returns
        -------
        self : object
            This estimator with its updated configuration.
        """
        return _ENTRY(self, params, _CONTEXT)


def collect():
    """Refresh `_CLASSES` with the classes of the loaded families."""
    # glue: fixed class inventory of the families imported so far
    global _CLASSES
    import sys
    found = []
    for name, source in _CLASS_SOURCES.items():  # glue: fixed vetted class table
        module = sys.modules.get(__package__ + '.' + source)
        if module is not None:
            found.append(getattr(module, name))
    _CLASSES = tuple(found)


def _install(classes):
    # glue: install only source-vetted classes before class-layout snapshots
    import inspect
    import sys
    from ._mode import ParamsMixin
    from ._native_public_attributes import _resolved
    from ._native_archive_bridge import connect as archive_connection
    archive = archive_connection()[1].get('host_archive_context')
    for cls in classes:  # glue: exact source-vetted full-reset constructor contracts
        if _resolved(cls, 'set_params', _CONTEXT['missing']) is not ParamsMixin.set_params:
            raise TypeError('full-reset setter must preserve its original ParamsMixin contract')
        signature = inspect.signature(cls.__init__)
        parameters = tuple(signature.parameters.values())[1:]
        if any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in parameters):  # glue: import-time signature admission
            raise TypeError('field-only setter requires an ordinary declared constructor signature')
        names = tuple(p.name for p in parameters)  # glue: exact parameter metadata, borrowed defaults
        events = _EVENTS[cls.__name__]
        if any(kind == 'parameter' and value not in names for _, kind, value, _ in events):  # glue: refuse stale constructor schemas
            raise TypeError('field-only setter schema differs from its constructor signature')
        # The original initializer that raises the argument errors: the
        # NumericModeMixin wrapper forwards to it without numeric_mode.
        original = _resolved(cls, '__init__', _CONTEXT['missing'])
        inner = getattr(original, '__wrapped__', original)
        wrapped = inner is not original
        if wrapped and 'numeric_mode' in inspect.signature(inner).parameters:
            # A positional numeric_mode would reach the inner initializer
            # while the wrapper's keyword stays None: not this schema.
            raise TypeError('native constructor requires the injected keyword-only numeric_mode')
        kwonly = tuple(p.name for p in inspect.signature(inner).parameters.values()  # glue: import-time constructor metadata
                       if p.kind is p.KEYWORD_ONLY and not (wrapped and p.name == 'numeric_mode'))
        _CONTEXT['plans'][cls] = dict(names=names, events=events, signature=signature,
            defaults={p.name: p.default for p in parameters if p.default is not p.empty},  # glue: exact default reference identities
            class_name=cls.__name__,
            validation_kind='umap' if cls.__name__ == 'UMAP' else None,
            positional=tuple(p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD),  # glue: native argument schema
            kwonly=kwonly,
            required_positional=tuple(p.name for p in parameters  # glue: native argument schema
                                      if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is p.empty),
            required_kwonly=tuple(p.name for p in parameters  # glue: native argument schema
                                  if p.kind is p.KEYWORD_ONLY and p.default is p.empty),
            rule_fields=tuple(dict.fromkeys(event[0] for event in events if event[3])),  # glue: numeric_mode rule destination fields
            label=getattr(inner.__code__, 'co_qualname', inner.__qualname__))
        _CONTEXT['originals'][cls] = original
    # The explicit host inference subclasses have their own constructor and
    # storage contracts. Keep their existing setter debt visible: changing a
    # selected base must not silently replace a host setter with a refusal.
    # The host module may load before or after its bases' families, so its
    # descendants of any installed base are kept on the original setter.
    host = sys.modules.get(__package__ + '._classical_host')
    previous = _CONTEXT.get('deferred_host_classes', ())
    installed = set(_CONTEXT['originals'])
    deferred = () if host is None else tuple(
        cls for cls in vars(host).values()  # glue: fixed built-in descendants outside this migration
        if type(cls) is type and cls not in _CONTEXT['plans'] and cls not in previous
        and cls.__module__ == host.__name__ and 'set_params' not in vars(cls)
        and any(base in installed for base in cls.__mro__[1:])  # glue: fixed host ancestry at connection
        and _resolved(cls, 'set_params', _CONTEXT['missing']) in (ParamsMixin.set_params,
                                                                   NativeFullResetParams.set_params))
    for cls in deferred:  # glue: preserve the original unconverted host contract before changing bases
        cls.set_params = ParamsMixin.set_params
    _CONTEXT['deferred_host_classes'] = previous + deferred
    # Validate the complete set before replacing any public setter.
    for cls in classes:  # glue: one direct native-call wrapper for the setter and the constructor
        cls.set_params = NativeFullResetParams.set_params
        original = _CONTEXT['originals'][cls]
        function = _initializer(cls)
        function.__name__ = '__init__'
        function.__qualname__ = cls.__qualname__ + '.__init__'
        function.__module__ = cls.__module__
        function.__doc__ = original.__doc__
        function.__signature__ = _CONTEXT['plans'][cls]['signature']
        cls.__init__ = function
        if cls.__name__ in _NATIVE_PUBLIC_LOADERS and archive is not None:
            archive.setdefault('host_archive_native_loader_hooks', {})[cls] = (
                _resolved(cls, '_mojolearn_after_fit', _CONTEXT['missing']),
                _CONTEXT['missing'])
        if cls.__name__ in _NATIVE_PUBLIC_CLASSES:
            cls._mojolearn_native_public = True


def install():
    # glue: the loaded NumPy-free classes not installed yet; _DEFERRED_MODULES
    # connect when they load
    _install(tuple(cls for cls in _CLASSES if cls not in _CONTEXT['originals']))
    from ._native_mode_reconstruction import install as install_reconstruction
    install_reconstruction(_CONTEXT, NativeFullResetParams.set_params)


def _connect(classes, parameter_metadata):
    # glue: final method/storage identities after native attribute installation
    import sys
    from ._native_public_attributes import _resolved, _archive_finalizers, _archive_finalizer_binding, _CONTEXT as attributes
    archives = _archive_finalizers(sys.modules[__package__])
    protected = ('__new__', '__init__', 'set_params', 'get_params', '_get_param_names',
                 '__getattribute__', '__getattr__', '__setattr__', '__delattr__', '__del__', '__dict__', '__slots__')
    for cls in classes:  # glue: final snapshots after native attribute installation
        _CONTEXT.setdefault('connected_classes', set()).add(cls)
        plan = _CONTEXT['plans'][cls]
        class_protected = protected + (('_parameters', '_extras') if plan['validation_kind'] == 'umap' else ())
        mro = type.__dict__['__mro__'].__get__(cls)
        fields = tuple(dict.fromkeys(event[0] for event in plan['events']))  # glue: constructor destinations in first-insertion order
        plan.update(mro=mro, protected=class_protected,
            layout_hooks=tuple((base, name, vars(base).get(name)) for base in mro for name in class_protected),  # glue: native storage and constructor identities
            inherited_hooks=tuple((name, _resolved(cls, name, _CONTEXT['missing'])) for name in class_protected),  # glue: passive subclass admission
            field_hooks=tuple((base, name, vars(base).get(name)) for base in mro for name in fields),  # glue: parameter/constant field descriptors
            field_names=fields)
        if _resolved(cls, '__new__', _CONTEXT['missing']) is not object.__new__:
            raise TypeError('full-reset native constructor requires ordinary object allocation')
        if _resolved(cls, '__setattr__', _CONTEXT['missing']) is not object.__setattr__:
            raise TypeError('full-reset native constructor cannot invoke Python assignment hooks')
        finalizer = _resolved(cls, '__del__', _CONTEXT['missing'])
        plan['release_allowed'] = (finalizer is _CONTEXT['missing']
            or _archive_finalizer_binding(cls, finalizer, attributes, archives) is not None)
        metadata = parameter_metadata['plans'][cls]
        if set(metadata['names']) != set(plan['names']):
            raise TypeError('full-reset native constructor and getter schemas differ')
        plan['constructor_hooks'] = tuple((name, _resolved(cls, name, _CONTEXT['missing']))  # glue: final lookup/assignment slots
                                          for name in _CONSTRUCTOR_HOOKS + (('_parameters', '_extras') if plan['validation_kind'] == 'umap' else ()))  # glue: fixed constructor protocols


def connect(parameter_metadata, linear, declared, estimator_context):
    # glue: final method/storage identities and already-supported native setters
    from ._native_public_attributes import _CONTEXT as attributes
    # Each class is snapshotted once, when its family first connects; a later
    # family load never re-reads an already connected class.
    done = _CONTEXT.setdefault('connected_classes', set())
    _connect(tuple(cls for cls in _CLASSES if cls not in done), parameter_metadata)
    from ._native_mode_reconstruction import finish as finish_reconstruction
    finish_reconstruction(_CONTEXT, parameter_metadata,
                          tuple(cls for cls in _CONTEXT.get('reconstruction_classes', ()) if cls not in done))
    native_setters = {}
    # Export identity and ABI are connection metadata, not transitive runtime
    # qualification. The nested ownership/callback tests must pass before
    # admitting this candidate; some inherited setters still owe that repair.
    for cls, plan in estimator_context['family_msel_estimators'].items():  # glue: existing built-in setter ABI connections
        recipe = plan['methods'].get('set_params')
        if recipe is not None and type(plan['entry']) is BuiltinFunctionType:
            if recipe[1] != (('arg', 0),):
                raise TypeError('nested setter requires the connected ordinary parameter-dictionary ABI')
            native_setters[cls] = dict(layout=plan, entry=plan['entry'], operation=recipe[0], context=plan['context'])
    # Native slot types a passive or host subclass may carry instead of the
    # owner's exact slot: object's own slots and the native attribute
    # descriptors. A Python function hook is never admitted.
    hook_types = [type(vars(object)['__getattribute__']), type(vars(object)['__setattr__'])]
    if 'descriptor' in attributes:
        hook_types.append(type(attributes['descriptor']))
    # A deferred module imported while the package was still importing was
    # installed then (setter and constructor) and seen by every package-level
    # snapshot; finish it here, before the constructors admit calls.
    for classes in _CONTEXT.pop('pending_classes', ()):  # glue: deferred modules loaded during package import
        _connect(classes, parameter_metadata)
    _CONTEXT.update(parameter_metadata=parameter_metadata, linear=linear, declared=declared,
                    native_setters=native_setters, constructor_hook_types=tuple(hook_types),
                    umap_validation_context=estimator_context, ready=True)


def connect_module(namespace):
    # glue: called once by each _DEFERRED_MODULES module as its last statement,
    # so its classes connect in the package order: native setter and
    # constructor, attribute slots, getter schemas, then the final full-reset
    # snapshot (which records the native __init__ and the constructor hooks).
    # The package's lazy name hook repeats the attribute and getter
    # connections as no-ops.
    import sys
    from . import _mode
    from ._native_public_attributes import connect_classes as connect_attributes
    from ._native_params_metadata import connect_classes as connect_parameters
    names = _DEFERRED_MODULES[namespace['__name__'].rpartition('.')[2]]
    classes = tuple(namespace[name] for name in names)  # glue: fixed class declarations of this module
    _install(classes)
    if not _CONTEXT['ready']:
        _CONTEXT.setdefault('pending_classes', []).append(classes)
        return
    package = sys.modules[__package__]
    connect_attributes(package)
    connect_parameters(_mode._PARAMS_METADATA_CONTEXT, package)
    _connect(classes, _CONTEXT['parameter_metadata'])
