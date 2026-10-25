# SPDX-License-Identifier: Apache-2.0
"""Import-time plans for the compiled public parallel-driver boundary."""
from types import BuiltinFunctionType, GetSetDescriptorType, ModuleType, SimpleNamespace


def _binding(name, mode):
    # glue: resolve lazy host bindings once, before a public operation.
    from . import _backend
    try:
        module = _backend.binding(name, mode)
        if type(module) is _backend._HostBinding:
            module = _backend.load_host_module(module._HostBinding__basename)
        if type(module) is not ModuleType:
            getattr(module, 'parallel_public_probe')
            raise ImportError('parallel driver requires a compiled binding')
        entries = {'__name__': module.__name__}
        for key, value in vars(module).items():  # glue: fixed compiled exports, no runtime module hooks
            if type(value) is BuiltinFunctionType:
                entries[key] = value
        return SimpleNamespace(**entries)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        return type(error).__name__ + ': ' + str(error)


def _class(cls):
    # glue: immutable class/storage identities read at import.
    mro = type.__dict__['__mro__'].__get__(cls)
    fields = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)  # glue: fixed class ancestry
    descriptor = next(row['__dict__'] for row in fields if '__dict__' in row)  # glue: native instance dictionary
    if type(cls) is not type or type(descriptor) is not GetSetDescriptorType:
        raise TypeError('native parallel drivers require ordinary class dictionaries')
    names = ('__dict__', '__getattribute__', '__getattr__', '__setattr__', '__del__', '__init__',
             'fit', 'predict', 'predict_proba', 'transform', '_bind')
    hooks = tuple((base, name, row.get(name)) for base, row in zip(mro, fields) for name in names)  # glue: frozen storage/method identities
    return dict(mro=mro, fields_get=descriptor.__get__, hooks=hooks,
                kind=getattr(cls, '_estimator_type', None), parameters=tuple(getattr(cls, '_parameter_names', ())))


def connect(context):
    # glue: no operation calls this function; each module connects at import.
    if 'parallel_public' in context:
        return
    from . import _backend, _native_family_bridge
    from . import linear_model as lm, decomposition as dec, _solver_impl as cd, cluster as cl
    from . import _svm_impl as svm, _gp_impl as gp, density as den, mixture as mix, hdbscan as hdb
    from . import _cholesky_impl as chol, kernel_methods as km, _hierarchy_impl as hier
    from . import _spectral_impl as spectral, _umap_impl as umap, _expansion_cluster as xc
    from . import ensemble as ens, _gbdt_adapters as boost, _iforest_impl as ifo, randomforest as rf
    from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    from . import _native_parallel_shard as shard
    family, _ = _native_family_bridge.connect()
    arg = lambda i: ('arg', i)
    const = lambda v: ('constant', v)
    bind = lambda n, mode='model': ('binding', n, mode)
    a, y = arg(0), arg(1)
    _backend.default_mode()
    plans, classes = {}, {}
    device_args = dict(fit_logistic=2, fit_gram_estimator=2, fit_kmeans=1,
        fit_coordinate_descent=2, fit_svm=2, predict_svm=1, fit_gaussian_process=2,
        predict_gaussian_process=1, fit_dbscan=1, fit_gaussian_mixture=1,
        predict_gaussian_mixture=1, fit_hdbscan=1, bootstrap=11, permutation_test=9,
        monte_carlo_integrate=7, fit_cholesky=1, solve_cholesky=1, fit_kernel_method=2,
        apply_kernel_method=1, transform_rbf_sampler=1, fit_graph=1, transform_umap=1,
        fit_boosting=2, fit_isolation_forest=1, score_isolation_forest=1,
        fit_ordered_rmse=3, fit_feature_freq=2)

    def add(name, cls, driver, operation, recipe, modules, *, entry=None, owner=None,
            mode='optional', check=None, output=None, method_arg=None, error=None, route='cooperative', numeric_mode_arg=None):
        # glue: the public call's fixed ABI, class identities and refusal text.
        key = cls if cls is not None else None
        plans.setdefault(name, {})[key] = dict(driver=driver, operation=operation, recipe=recipe,
            modules=modules, entry=entry, owner=owner, mode=mode, check=check, output=output, numeric_mode_arg=numeric_mode_arg,
            method_arg=method_arg, device_arg=device_args[name], error=error or ('requires mojolearn.' + cls.__name__ if cls else 'invalid parallel operation'), route=route)
        if cls is not None and cls not in classes:
            classes[cls] = _class(cls)

    add('fit_logistic', lm.LogisticRegression, 'glm_fit', 'fit', (a,y,const(None)), ('_mojolearn_estimators',), entry=lm._LR_CALL, owner=lm._LR_CONTEXT)
    for cls, entry, op, owner, arity in (  # glue: fixed Gram owner alternatives
        (lm.LinearRegression,lm._native_classical,'ols_fit',lm._CLASSICAL,3),
        (lm.Ridge,lm._native_classical,'ridge_fit',lm._CLASSICAL,3),
        (dec.PCA,dec._PCA_ENTRY,'fit',dec._PCA_CONTEXT,2),
        (dec.TruncatedSVD,dec._TSVD_ENTRY,'fit',dec._TSVD_CONTEXT,2),
    ):
        add('fit_gram_estimator',cls,'gram_fit',op,(a,y,arg(3))[:arity],
            ('_mojolearn_estimators','_mojolearn_x_linear','_mojolearn_x_decomp'),entry=entry,owner=owner,check='gram',
            error='fit_gram_estimator requires LinearRegression, Ridge, PCA or TruncatedSVD')
    add('fit_kmeans',cl.KMeans,'kmeans_fit','family_kmeans_fit',(bind(cl.KMeans._BINDING),a,arg(2)),
        ('_mojolearn','_mojolearn_x_cluster'),check='kmeans')
    for cls in (cd.Lasso,cd.ElasticNet):  # glue: the fixed coordinate-descent classes
        add('fit_coordinate_descent',cls,'solver_fit','cd_fit',(a,y,const(None)),('_mojolearn_solver',),entry=cd._CD_CALL,owner=cd._CD_CONTEXT,mode='default')
    for cls,prefix in ((svm.SVC,'svc'),(svm.SVR,'svr')):  # glue: fixed SVM owner operations
        add('fit_svm',cls,'svm_fit','family_svc_'+prefix+'_fit',(bind(svm._EXT_NAME),a,y,arg(3)),(svm._EXT_NAME,))
        choices={'predict':'family_svc_'+prefix+'_predict'}
        if cls is svm.SVC:
            choices['decision_function']='family_svc_svc_decision'
        add('predict_svm',cls,'svm_predict',choices,(bind(svm._EXT_NAME),a),(svm._EXT_NAME,),method_arg=2,output='argument')
    add('fit_gaussian_process',gp.GaussianProcessRegressor,'gp_fit','family_gp_gpr_fit',(bind('_mojolearn_gp'),a,y),('_mojolearn_gp',))
    add('predict_gaussian_process',gp.GaussianProcessRegressor,'gp_predict','family_gp_gpr_predict',
        (bind('_mojolearn_gp'),a,arg(2),const(False)),('_mojolearn_gp',),check='posterior',output='predict')
    add('fit_dbscan',den.DBSCAN,'dbscan_fit','family_cfb_dbscan_fit',(bind('_mojolearn_estimators'),a,arg(2)),('_mojolearn_estimators',))
    add('fit_gaussian_mixture',mix.GaussianMixture,'gmm_fit','family_mix_gmm_fit',
        (bind('_mojolearn_mixture'),bind('_mojolearn_x_cluster'),a),('_mojolearn_mixture','_mojolearn_x_cluster','_mojolearn'),check='matrix')
    add('predict_gaussian_mixture',mix.GaussianMixture,'gmm_predict',
        {name:'family_mix_gmm_'+name for name in ('predict','predict_proba','score_samples')},  # glue: fixed output ABI names
        (bind('_mojolearn_mixture'),a),('_mojolearn_mixture',),method_arg=2,check='gmm_predict',output='argument')
    add('fit_hdbscan',hdb.HDBSCAN,'hdbscan_fit','family_cfa_hdbscan_fit',(bind(hdb.HDBSCAN._BINDING),a),('_mojolearn_hdbscan','_mojolearn_solver'),check='matrix')
    # Resampling argument tuples include devices last and numeric_mode before it.
    add('bootstrap',None,'resample','family_rs_bootstrap',
        (bind('_mojolearn_resample','identical'),)+tuple(arg(i) for i in range(9))+(const('identical'),arg(10)),  # glue: fixed bootstrap ABI
        ('_mojolearn_resample',),check='resample', numeric_mode_arg=9)
    add('permutation_test',None,'resample','family_rs_permutation_test',
        (bind('_mojolearn_resample','identical'),)+tuple(arg(i) for i in range(7))+(const('identical'),arg(8)),  # glue: fixed permutation ABI
        ('_mojolearn_resample',),check='resample', numeric_mode_arg=7)
    add('monte_carlo_integrate',None,'resample','family_rs_monte_carlo_integrate',
        (bind('_mojolearn_resample','identical'),)+tuple(arg(i) for i in range(6))+(const('identical'),),  # glue: fixed Monte Carlo ABI
        ('_mojolearn_resample',),check='resample', numeric_mode_arg=6)
    chol_binding = chol._CPU_BINDING if _backend._CPU_ONLY is not None else chol._GPU_BINDING
    for name,op in (('fit_cholesky','family_chol_par_fit'),('solve_cholesky','family_chol_par_solve')):  # glue: fixed Cholesky operations
        add(name,chol.Cholesky,None,op,(bind(chol_binding),a,arg(1)),(),mode=None,route='direct')
    for cls in (km.KernelRidge,km.Nystroem):  # glue: fixed kernel-method operations
        add('fit_kernel_method',cls,None,'family_km_par_fit',(bind(cls._BINDING),a,y,arg(2)),(),mode=None,route='direct')
        add('apply_kernel_method',cls,None,'family_km_par_apply',(bind(cls._BINDING),a,arg(1)),(),mode=None,route='direct',output='predict' if cls is km.KernelRidge else 'transform')
    add('transform_rbf_sampler',km.RBFSampler,None,'transform_rbf_sampler',
        (('model',),a,arg(1),arg(2),bind(km.RBFSampler._BINDING)),(),mode=None,route='shard',output='transform')
    graph_modules=('_mojolearn_metrics','_mojolearn','_mojolearn_estimators','_mojolearn_solver','_mojolearn_x_cluster','_mojolearn_x_neighbors','_mojolearn_x_decomp','_mojolearn_hdbscan')
    add('fit_graph',hier.AgglomerativeClustering,'graph_fit','family_cfb_agglo_fit',
        (bind(hier.AgglomerativeClustering._BINDING),bind('_mojolearn_solver','identical'),bind(xc._XCluster._BINDING),bind('_mojolearn_x_neighbors'),a),graph_modules,mode='effective')
    add('fit_graph',spectral.SpectralClustering,'graph_fit','family_cfb_sc_fit',
        (bind('_mojolearn_metrics','default'),bind('_mojolearn_x_decomp','default'),bind('_mojolearn','identical'),bind('_mojolearn_x_cluster'),a),graph_modules,mode='effective')
    add('fit_graph',umap.UMAP,'graph_fit','family_mani_umap_fit',(bind(umap.UMAP._BINDING),a,const(None)),graph_modules,mode='effective')
    add('transform_umap',umap.UMAP,'umap_transform','family_mani_umap_transform',
        (bind(umap.UMAP._BINDING,'identical'),a),graph_modules,mode=None,check='umap_transform',output='transform')
    for cls,op,arity in ((ens.GradientBoosting,'trees_gbdt_fit',8),  # glue: fixed boosting public ABI arities
                        (boost.GradientBoostingClassifier,'trees_gbdt_adapter_classifier_fit',4),
                        (boost.GradientBoostingRegressor,'trees_gbdt_adapter_regressor_fit',4)):
        add('fit_boosting',cls,'gbdt_fit',op,(a,y,arg(3),arg(4),const(None),const(None),const(None),const(None))[:arity],
            ('_mojolearn_gbdt',),entry=ens._TREES if cls is ens.GradientBoosting else boost._TREES,
            owner=ens._TREES_CONTEXT if cls is ens.GradientBoosting else boost._TREES_CONTEXT)
    add('fit_isolation_forest',ifo.IsolationForest,'iforest_fit','trees_iforest_fit',(a,const(None)),
        ('_mojolearn_svm',),entry=ifo._TREES,owner=ifo._TREES_CONTEXT,check='iforest')
    add('score_isolation_forest',ifo.IsolationForest,'iforest_score',
        {'score_samples':'trees_iforest_score_samples','decision_function':'trees_iforest_decision_function','predict':'trees_iforest_predict'},
        (a,),('_mojolearn_svm',),entry=ifo._TREES,owner=ifo._TREES_CONTEXT,method_arg=2,output='argument')
    add('fit_ordered_rmse',ens.OrderedRMSE,'ordered_rmse_fit','trees_gbdt_ordered_fit',(a,y,arg(2),arg(4)),
        ('_mojolearn_gbdt',),entry=ens._TREES,owner=ens._TREES_CONTEXT)
    add('fit_feature_freq',ens.ExperimentalTwoLevelFeatureFreq,'gbdt_fit','trees_gbdt_feature_freq_fit',(a,y,arg(3)),
        ('_mojolearn_gbdt',),entry=ens._TREES,owner=ens._TREES_CONTEXT)
    forest_classes=(rf.RandomForestClassifier,rf.RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor)
    for cls in forest_classes:  # glue: native predictor snapshot layout
        classes[cls]=_class(cls)
    basenames={item[1] for group in plans.values() for spec in group.values() for item in spec['recipe'] if item[0]=='binding'}  # glue: fixed binding names
    basenames.update(name for group in plans.values() for spec in group.values() for name in spec['modules'])  # glue: fixed plan bindings
    basenames.update(('_mojolearn_rf','_mojolearn_trees'))
    modules={name:{mode:_binding(name,mode) for mode in ('identical','fast')} for name in sorted(basenames)}  # glue: import-time binding connection
    context['parallel_public']=dict(plans=plans,classes=classes,modules=modules,family=family,
        shard=shard._native_parallel_shard,backend=vars(_backend),cpu_only=_backend._CPU_ONLY is not None,
        pca=dec.PCA,dense_solvers=tuple(dec.PCA._DENSE_SOLVERS),decompositions=(dec.PCA,dec.TruncatedSVD),
        forests=forest_classes,forest_classifier=(rf.RandomForestClassifier,ExtraTreesClassifier),
        trees_entry=rf._TREES,trees_context=rf._TREES_CONTEXT)
    shard.register(context=context,rbf_class=km.RBFSampler)


def register_predictor(cls, context):
    # glue: a public class connects after its definition, never at construction.
    import _thread
    connect(context)
    meta=context['parallel_public']
    meta.update(predictor_class=cls,predictor_layout=_class(cls),lock_type=_thread.RLock)
