# SPDX-License-Identifier: Apache-2.0
"""Import-time recipes for native reconstruction by inherited set_params.

Only fixed built-in native initializer ABIs are connected. No constructor body,
user function, signature binder or importer is invoked by product execution.
"""
from types import BuiltinFunctionType, FunctionType, ModuleType

# Source-vetted complete native constructor calls; parameter identities/order
# come from the original bodies. Inherited passive classes reuse the recipe.
_NATIVE_RECIPES = {
    '_arima_impl.ARIMA': ('_TSA', 'arima_initialize', (('parameter', 'order'), ('parameter', 'seasonal_order'), ('parameter', 'trend'), ('parameter', 'method'), ('parameter', 'maxiter'), ('parameter', 'verbose'), ('parameter', 'output_type'), ('parameter', 'numeric_mode')), '_TSA_CONTEXT'),
    '_cholesky_impl.Cholesky': ('_NATIVE', 'family_chol_initialize', (('parameter', 'jitter'), ('parameter', 'numeric_mode')), '_NATIVE_CONTEXT'),
    'linear_model.QNRegressor': ('_QN_CALL', 'initialize', (('parameter', 'loss'), ('parameter', 'fit_intercept'), ('parameter', 'l1_strength'), ('parameter', 'l2_strength'), ('parameter', 'max_iter'), ('parameter', 'tol'), ('parameter', 'delta'), ('parameter', 'linesearch_max_iter'), ('parameter', 'lbfgs_memory'), ('parameter', 'warm_start'), ('parameter', 'penalty_normalized'), ('parameter', 'numeric_mode')), '_QN_CONTEXT'),
    'ensemble.GradientBoosting': ('_TREES', 'trees_gbdt_init', (('parameter', 'loss'), ('parameter', 'n_estimators'), ('parameter', 'max_depth'), ('parameter', 'learning_rate'), ('parameter', 'l2_leaf_reg'), ('parameter', 'border_count'), ('parameter', 'random_state'), ('parameter', 'loss_alpha'), ('parameter', 'loss_q'), ('parameter', 'loss_delta'), ('parameter', 'loss_variance_power'), ('parameter', 'loss_border'), ('parameter', 'leaf_estimation_method'), ('parameter', 'leaf_estimation_iterations'), ('parameter', 'bootstrap_type'), ('parameter', 'bagging_temperature'), ('parameter', 'subsample'), ('parameter', 'cat_features'), ('parameter', 'one_hot_features'), ('parameter', 'od_type'), ('parameter', 'od_pvalue'), ('parameter', 'od_wait'), ('parameter', 'use_best_model'), ('parameter', 'best_model_min_trees'), ('parameter', 'score_function'), ('parameter', 'nan_mode'), ('parameter', 'random_strength'), ('parameter', 'use_pointwise_searcher'), ('parameter', 'boost_from_average'), ('parameter', 'border_build_max_samples'), ('parameter', 'class_weights'), ('parameter', 'permutation_count'), ('parameter', 'ctr_estimation_permutation_id'), ('parameter', 'grow_policy'), ('parameter', 'max_leaves'), ('parameter', 'min_data_in_leaf'), ('parameter', 'min_split_gain'), ('parameter', 'min_child_hessian'), ('parameter', 'feature_fraction'), ('parameter', 'feature_border_type'), ('parameter', 'boosting_type'), ('parameter', 'fold_len_multiplier'), ('parameter', 'fold_permutation_block'), ('parameter', 'numeric_mode')), '_TREES_CONTEXT'),
    '_x_sequence_theta.AutoTheta': ('_XSEQ', 'theta_initialize', (('parameter', 'season_length'), ('parameter', 'decomposition_type'), ('parameter', 'model'), ('parameter', 'initial_smoothed'), ('parameter', 'alpha'), ('parameter', 'theta'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_ets.ETS': ('_XSEQ', 'ets_initialize', (('parameter', 'season_length'), ('parameter', 'model'), ('parameter', 'damped'), ('parameter', 'alpha'), ('parameter', 'beta'), ('parameter', 'phi'), ('parameter', 'gamma'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_ets.DampedETS': ('_XSEQ', 'ets_initialize', (('parameter', 'season_length'), ('parameter', 'error'), ('parameter', 'alpha'), ('parameter', 'beta'), ('parameter', 'phi'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_garch.GARCH': ('_XSEQ', 'garch_initialize', (('parameter', 'p'), ('parameter', 'o'), ('parameter', 'q'), ('parameter', 'mean'), ('parameter', 'dist'), ('parameter', 'power'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_prophet.ProphetForecaster': ('_XSEQ', 'prophet_initialize', (('parameter', 'growth'), ('parameter', 'changepoints'), ('parameter', 'n_changepoints'), ('parameter', 'changepoint_range'), ('parameter', 'yearly_seasonality'), ('parameter', 'weekly_seasonality'), ('parameter', 'daily_seasonality'), ('parameter', 'seasonality_mode'), ('parameter', 'seasonality_prior_scale'), ('parameter', 'holidays_prior_scale'), ('parameter', 'changepoint_prior_scale'), ('parameter', 'max_iter'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_stl.STL': ('_XSEQ', 'stl_initialize', (('parameter', 'endog'), ('parameter', 'period'), ('parameter', 'seasonal'), ('parameter', 'trend'), ('parameter', 'low_pass'), ('parameter', 'seasonal_deg'), ('parameter', 'trend_deg'), ('parameter', 'low_pass_deg'), ('parameter', 'robust'), ('parameter', 'seasonal_jump'), ('parameter', 'trend_jump'), ('parameter', 'low_pass_jump'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_var.VAR': ('_XSEQ', 'var_initialize', (('parameter', 'endog'), ('parameter', 'exog'), ('parameter', 'numeric_mode')), '_XSEQ_CONTEXT'),
    '_x_sequence_autoarima.AutoARIMA': ('_TSA', 'autoarima_initialize', (('parameter', 'endog'),), '_TSA_CONTEXT'),
    '_classical_host.HostKernelDensity': ('_native_classical', 'host_density_initialize', (('constant', 'kde'), ('parameter', 'bandwidth'), ('parameter', 'kernel'), ('parameter', 'metric'), ('parameter', 'algorithm'), ('parameter', 'atol'), ('parameter', 'rtol'), ('parameter', 'breadth_first'), ('parameter', 'leaf_size'), ('parameter', 'metric_params'), ('parameter', 'numeric_mode')), '_CLASSICAL'),
    '_classical_host.HostIsolationForest': ('_native_classical', 'host_density_initialize', (('constant', 'iforest'), ('parameter', 'n_estimators'), ('parameter', 'max_samples'), ('parameter', 'max_depth'), ('parameter', 'max_features'), ('parameter', 'bootstrap'), ('parameter', 'contamination'), ('parameter', 'random_state'), ('parameter', 'warm_start'), ('parameter', 'n_jobs'), ('parameter', 'verbose'), ('parameter', 'output_type'), ('parameter', 'numeric_mode')), '_CLASSICAL'),
}


def install(context, setter, modules=None):
    # glue: connect loaded product declarations; lazy modules call this after
    # definition and before their attribute/parameter snapshots are frozen.
    import inspect
    import sys
    from . import _mode
    from ._native_public_attributes import _resolved
    selected = tuple(sys.modules.items()) if modules is None else tuple(modules)
    added = []
    table = context.setdefault('reconstruction_classes', [])
    pending = context.setdefault('reconstruction_pending', {})
    # Freeze candidate identities before replacing any base-class setter;
    # a later subclass can have a different constructor despite inheritance.
    candidates = []
    for module_name, module in selected:  # glue: loaded module/class metadata only
        if (type(module) is not ModuleType or not module_name.startswith(__package__ + '.')
                or module_name.startswith(__package__ + '.tests')):
            continue
        for cls in tuple(vars(module).values()):  # glue: exact built-in class identities
            if (type(cls) is not type or cls in context['plans'] or cls in pending
                    or vars(cls).get('__module__') != module_name):
                continue
            if _resolved(cls, 'set_params', context['missing']) is _mode.ParamsMixin.set_params:
                if cls not in candidates:
                    candidates.append(cls)
    for cls in candidates:  # glue: each complete native reconstruction schema
        module_name = cls.__module__
        ancestry = type.__dict__['__mro__'].__get__(cls)
        # Already source-vetted field-only constructors can be replayed for
        # the explicit CPU subclasses without executing their Python shell.
        inherited = next((context['plans'][base] for base in ancestry[1:]  # glue: native ancestor constructor schemas
                          if base in context['plans'] and 'native_constructor' not in context['plans'][base]), None)
        initializer = _resolved(cls, '__init__', context['missing'])
        if inherited is not None and module_name == __package__ + '._classical_host' and '__init__' not in vars(cls):
            plan = dict(inherited)
            plan['class_name'] = cls.__name__
            context['plans'][cls] = plan
        else:
            owner = next((base for base in ancestry  # glue: fixed native constructor recipe owner
                          if base.__module__.removeprefix(__package__ + '.') + '.' + base.__name__ in _NATIVE_RECIPES), None)
            if owner is None:
                pending[cls] = 'no connected complete native constructor recipe'
                continue
            owner_module = sys.modules[owner.__module__]
            recipe = _NATIVE_RECIPES[owner.__module__.removeprefix(__package__ + '.') + '.' + owner.__name__]
            entry = vars(owner_module).get(recipe[0])
            if type(entry) is not BuiltinFunctionType:
                pending[cls] = 'native constructor entry is unavailable'
                continue
            # A passive descendant must still inherit this constructor;
            # class-specific Python initialization is not a native recipe.
            actual = inspect.unwrap(initializer)
            expected = inspect.unwrap(vars(owner)['__init__'])
            if type(actual) is not FunctionType or actual is not expected:
                pending[cls] = 'constructor differs from the connected native recipe'
                continue
            signature = inspect.signature(initializer)
            parameters = tuple(signature.parameters.values())[1:]
            if any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in parameters):  # glue: ordinary native constructor argument words
                pending[cls] = 'constructor requires an explicit variadic native recipe'
                continue
            names = tuple(p.name for p in parameters)  # glue: fixed parameter metadata
            if any(kind == 'parameter' and value not in names for kind, value in recipe[2]):  # glue: refuse stale native ABI schemas
                raise TypeError('native reconstruction recipe differs from its constructor signature')
            post_mode = initializer is not actual and 'numeric_mode' in names
            context['plans'][cls] = dict(names=names, class_name=cls.__name__,
                events=tuple((name, 'parameter', name, False) for name in names),  # glue: guarded constructor fields
                validation_kind=None, native_constructor=dict(entry=entry, operation=recipe[1],
                    arguments=recipe[2], context=vars(owner_module)[recipe[3]], post_mode=post_mode))
        table.append(cls)
        added.append(cls)
        pending.pop(cls, None)
    for cls in tuple(candidates) + tuple(pending):  # glue: keep pending descendant contracts before changing a base
        if cls not in added and 'set_params' not in vars(cls):
            if any(base in added for base in cls.__mro__[1:]):  # glue: source-connected native ancestry
                cls.set_params = _mode.ParamsMixin.set_params
    for cls in added:  # glue: publish only completed native reconstruction schemas
        cls.set_params = setter
    return tuple(added)


def finish(context, parameter_metadata, classes):
    # glue: final snapshots for connected recipes. A class whose getter schema
    # or storage layout cannot be admitted keeps its original visible setter
    # instead of a native setter that would refuse every call.
    from . import _mode
    from ._native_full_reset_params import _connect
    table = context.setdefault('reconstruction_classes', [])
    pending = context.setdefault('reconstruction_pending', {})
    for cls in tuple(classes):  # glue: one import-time admission per connected class
        try:
            if cls not in parameter_metadata['plans']:
                raise TypeError('no connected native parameter getter schema')
            _connect((cls,), parameter_metadata)
        except TypeError as error:
            context['plans'].pop(cls, None)
            if cls in table:
                table.remove(cls)
            cls.set_params = _mode.ParamsMixin.set_params
            pending[cls] = str(error)


def connect_lazy(module, context, setter):
    # glue: complete the same install order for a newly imported sequence
    # module; its own imports (an ARIMA under AutoARIMA) connect with it.
    import sys
    from . import _mode
    from ._native_public_attributes import connect_classes as connect_attributes
    from ._native_params_metadata import connect_classes as connect_parameters
    added = install(context, setter)
    if added and context['ready']:
        package = sys.modules[__package__]
        connect_attributes(package)
        connect_parameters(_mode._PARAMS_METADATA_CONTEXT, package)
        finish(context, context['parameter_metadata'], added)
