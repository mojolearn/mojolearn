# SPDX-License-Identifier: Apache-2.0
"""Import-time recipes for prediction from every registered host archive.

`host_predict(path, X)` and `host_predict_proba(path, X)` are one native call:
the archive reader rebuilds the saved model and the compiled owner entry its
public method reaches runs from the fixed recipe below
(`bindings/host_archive_predict_native.mojo`). Nothing here runs from a
public prediction operation."""

# The saved classes whose public method selected its host binding through
# `_HostBound._bind` (or the GP classifier's own `_bind`), and therefore
# refused a non-identical saved numeric mode before reading X.
_MODE_CHECKED = frozenset((
    'HostGaussianMixture', 'HostGaussianProcessRegressor', 'HostGaussianProcessClassifier',
    'HostDBSCAN', 'HostAgglomerativeClustering', 'HostSpectralClustering',
    'HostKNeighborsClassifier', 'HostKNeighborsRegressor',
))

# Saved classes whose public interface has neither predict nor
# predict_proba: their AttributeError is preserved, never narrowed.
_NO_PREDICTION = ('HostCholesky', 'HostEmbedding', 'HostHDBSCAN', 'HostIVFIndex',
                  'HostKernelDensity', 'HostMinMaxScaler', 'HostNearestNeighbors',
                  'HostNystroem', 'HostPCA', 'HostRBFSampler', 'HostRadiusNeighbors',
                  'HostStandardScaler', 'HostTruncatedSVD', 'HostUMAP')


def connect(context, host, bindings):
    # glue: fixed class/operation metadata; this function never runs from a
    # public prediction operation, and no public method is a runtime callback.
    # `bindings` maps each host binding name to its loaded module or to the
    # recorded refusal text (`_classical_host._host_archive_prediction_bindings`).
    import inspect
    from types import BuiltinFunctionType, ModuleType
    from . import _forest_host, _gbdt_host
    from . import mixture, density, _hierarchy_impl, _spectral_impl, _gp_impl, _gpc_impl
    from . import _arima_impl, _tsa_impl

    archive = context['host_archive_context']
    compiled = host['_native_classical']

    def call(entry, operation, owner, *arguments):
        # glue: arguments are immutable (input-index, constant) recipe words.
        if type(entry) is not BuiltinFunctionType:
            entry = 'ImportError: native host archive prediction owner is unavailable'
        return (entry, operation, owner, arguments)

    x = ('input', 0)

    def constant(value):
        return ('constant', value)

    recipes = {}
    for name in ('HostLinearRegression', 'HostRidge'):  # glue: the two fixed linear schemas
        # The old nested public call was package-internal and returned its
        # native float32 result. Preserve that explicitly, not by frame depth.
        recipes[name] = (call(compiled, 'predict_score', context, x), None)
    for name in ('HostElasticNet', 'HostLasso', 'HostKernelRidge', 'HostKMeans',  # glue: complete native host inference families
                 'HostIsolationForest', 'HostSVR', 'HostSVC', 'HostLogisticRegression'):
        probability = name in ('HostSVC', 'HostLogisticRegression')
        recipes[name] = (call(compiled, 'archive_infer_predict', context, x),
                         call(compiled, 'archive_infer_predict_proba', context, x) if probability else None)

    for name in ('HostForest', 'HostGBDT'):  # glue: the two fixed tree archive owners
        recipes[name] = (call(compiled, 'host_tree_predict', context, x),
                         call(compiled, 'host_tree_proba', context, x))

    core = bindings['_mojolearn_core_host']
    for name, operation in (('HostKNeighborsClassifier', 'archive_knn_classifier'),  # glue: fixed whole-prediction KNN operations
                            ('HostKNeighborsRegressor', 'archive_knn_regressor')):
        recipes[name] = (call(compiled, operation, context, x, constant(False)),
                         call(compiled, operation, context, x, constant(True))
                         if name == 'HostKNeighborsClassifier' else None)

    mix = constant(bindings['mixture'])
    recipes['HostGaussianMixture'] = (
        call(mixture._NATIVE, 'family_mix_gmm_predict', mixture._NATIVE_CONTEXT, mix, x),
        call(mixture._NATIVE, 'family_mix_gmm_predict_proba', mixture._NATIVE_CONTEXT, mix, x))
    gp = constant(bindings['gp'])
    recipes['HostGaussianProcessRegressor'] = (
        call(_gp_impl._NATIVE, 'family_gp_gpr_predict', _gp_impl._NATIVE_CONTEXT,
             gp, x, constant(False), constant(False)), None)
    gpc = constant(bindings['gpc'])
    recipes['HostGaussianProcessClassifier'] = (
        call(_gpc_impl._NATIVE, 'family_gp_gpc_predict', _gpc_impl._NATIVE_CONTEXT,
             gpc, constant(bindings['gpc_core']), x),
        call(_gpc_impl._NATIVE, 'family_gp_gpc_predict_proba', _gpc_impl._NATIVE_CONTEXT, gpc, x))

    estimators = constant(bindings['_mojolearn_estimators_host'])
    spectral = constant(bindings['spectral'])
    for name, module, operation, kernel in (  # glue: the three saved clustering prediction recipes
            ('HostDBSCAN', density, 'family_cfb_dbscan_predict', estimators),
            ('HostAgglomerativeClustering', _hierarchy_impl, 'family_cfb_agglo_predict', estimators),
            ('HostSpectralClustering', _spectral_impl, 'family_cfb_sc_predict', spectral)):
        recipes[name] = (call(module._NATIVE, operation, module._NATIVE_CONTEXT, kernel, x), None)

    # Here X is the positional start argument, not a data matrix. The public
    # forecast methods' defaults remain (end=None, exog/index=None).
    recipes['HostARIMA'] = (call(_arima_impl._TSA, 'arima_predict', _arima_impl._TSA_CONTEXT,
                                 x, constant(None), constant(None)), None)
    recipes['HostExponentialSmoothing'] = (
        call(_tsa_impl._TSA, 'hw_predict', _tsa_impl._TSA_CONTEXT, x, constant(None), constant(None)), None)

    known = dict(host)
    known.update(HostForest=_forest_host.HostForest, HostGBDT=_gbdt_host.HostGBDT)
    for name in _NO_PREDICTION:  # glue: preserve the explicit absent-method schemas
        recipes[name] = (None, None)

    records = []
    covered = set()
    for classes in archive['host_archive_formats'].values():  # glue: complete saved-format inventory at package setup
        for cls in classes.values():  # glue: each fixed native class is recorded only once
            if cls in covered:
                continue
            covered.add(cls)
            name = cls.__name__
            if known.get(name) is not cls or name not in recipes:
                raise TypeError('saved host prediction requires an explicit native schema for ' + name)
            methods = tuple(inspect.getattr_static(cls, method, None)  # glue: retain descriptors as metadata without invoking them
                            for method in ('predict', 'predict_proba'))  # glue: fixed pair of method descriptors retained only at setup
            for descriptor, recipe in zip(methods, recipes[name]):  # glue: missing methods remain missing rather than narrowing available ones
                if (descriptor is None) != (recipe is None):
                    raise TypeError('saved host prediction schema does not match ' + name)
            records.append((cls, cls.__mro__, name, methods, recipes[name]))

    snapshots = {}
    for module in bindings.values():  # glue: imported host modules and their native exports are fixed ABI metadata
        if type(module) is str or module in snapshots:
            continue
        if type(module) is not ModuleType:
            raise TypeError('host archive bindings require exact native modules')
        fields = vars(module)
        if '__getattr__' in fields:
            raise TypeError('host archive bindings cannot retain Python module hooks')
        snapshots[module] = tuple((key, value) for key, value in fields.items()  # glue: retain actual compiled ABI export identities at setup
                                  if type(key) is str and type(value) is BuiltinFunctionType)
    archive.update(host_archive_predict_records=tuple(records),
                   host_archive_predict_modules=snapshots,
                   host_archive_predict_module_type=ModuleType,
                   host_archive_predict_entry_type=BuiltinFunctionType,
                   host_archive_predict_knn_binding=core,
                   host_archive_predict_mode_checked=_MODE_CHECKED,
                   host_archive_predict_svc_context=context['host_inference_context'],
                   host_archive_predict_classes=tuple(covered))
    context['host_archive_predict_context'] = archive

    # glue: inherited clustering methods use native public lifecycle owners.
    # Bind their saved Host classes to these exact CPU modules, independently
    # of the installation's selected GPU mode. Runtime never calls _HostBound._bind.
    from ._native_family_bridge import shared_context
    host_routes = []
    for name, family, module, reader in (  # glue: three fixed CPU owner connections at import
            ('HostDBSCAN', '_mojolearn_estimators', bindings['_mojolearn_estimators_host'], 'estimators_numeric_mode'),
            ('HostAgglomerativeClustering', '_mojolearn_estimators', bindings['_mojolearn_estimators_host'], 'estimators_numeric_mode'),
            ('HostSpectralClustering', '_mojolearn_metrics', bindings['spectral'], 'metrics_numeric_mode')):
        host_routes.append((host[name], family, module,
                            snapshots[module] if type(module) is ModuleType else (), reader))
    shared_context()['family_host_binding_connections'] = tuple(host_routes)
