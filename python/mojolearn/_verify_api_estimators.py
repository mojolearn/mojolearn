"""Eight-row endpoint fixtures; no timings, accuracy targets or datasets."""
__mojolearn_user_boundary__ = True

from ._verify_api_metadata import factory

METHODS = frozenset(('score', 'fit_transform', 'fit_predict', 'inverse_transform',
    'predict_joint_log_proba', 'predict_log_proba', 'predict_proba', 'decision_function',
    'get_precision', 'score_samples', 'get_feature_names_out', 'aic', 'bic',
    'reconstruction_error', 'latent_mean_and_variance', 'log_marginal_likelihood',
    'partial_fit', '__getstate__', '__setstate__', 'apply', 'get_leaf_values',
    'get_tree_leaf_counts', 'predict_with_leaves', 'staged_predict', 'predict_classes',
    'get_depth', 'classes_:get', 'n_candidate_distances_', 'used_query_tile_',
    'estimators_:get', 'estimators_:set', 'fit', 'transform'))

# Neural/stateful programs and raw arrays have their own tiny shape contracts.
SEPARATE = ('Adam', 'SGD', 'Adagrad', 'Adamax', 'Adafactor', 'Lion', 'LAMB', 'RMSprop',
            'NAdam', 'Mamba', 'Transformer', 'Sequential', 'MoE', 'Embedding', 'HostForest')


def applicable(public, methods):
    name = public.rsplit('.', 1)[-1]
    return (bool(set(methods) & METHODS) and not name.startswith(SEPARATE)
            and '.importers.' not in public and '.models.' not in public
            and 'Trainer' not in name and name not in ('Array', 'ExponentialSmoothing')
            and not any(n in name for n in ('Pool', 'BasicBlock', 'Dropout', 'BatchNorm')))


def dataset(public):
    import numpy as np
    # Fixed rational data avoids random input generation and rounding ambiguity.
    X = np.array([[1,2,3,1],[2,1,1,3],[3,4,2,2],[4,2,4,1],
                  [5,3,1,4],[6,1,3,2],[7,4,4,3],[8,2,2,4]], np.float32) / 8
    y = np.array([0,1,0,1,0,1,0,1], np.int64)
    if public.endswith('CategoricalNB'):
        X = (X*8).astype(np.int64)%3
    if public.endswith('ExperimentalTwoLevelFeatureFreq'):
        X = (X*8)%3
    return X, y, np.array([1,1.25,1.5,1.125,1.75,1.375,2,1.625], np.float32)


def estimator_checks(recorder, public, methods):
    import numpy as np
    obj = None
    fitted = False
    X, labels, target = dataset(public)
    name = public.rsplit('.', 1)[-1]
    if name=='NearestNeighbors':X=X[:,:2]

    def model():
        nonlocal obj
        if obj is None:
            overrides = {}
            if name=='NearestNeighbors':overrides['algorithm']='rbc'
            if name in ('LinearSVC', 'LinearSVR', 'SVC', 'SVR'):
                overrides['max_iter'] = 2
            if name in ('LocalOutlierFactor',):
                overrides['novelty'] = True
            if name == 'CalibratedClassifierCV':
                from . import LogisticRegression
                overrides.update(estimator=LogisticRegression(max_iter=2), cv=2)
            if name in ('GridSearchCV','RandomizedSearchCV'):
                from . import LinearDiscriminantAnalysis
                overrides.update(estimator=LinearDiscriminantAnalysis(n_components=1),
                    **{'param_distributions' if name=='RandomizedSearchCV' else 'param_grid': {'n_components':[1]}})
            if name=='RFE':
                from . import LogisticRegression
                overrides['estimator']=LogisticRegression(max_iter=2)
            if name=='GradientBoosting':
                overrides['loss']='Logloss'
            if name == 'RandomizedSearchCV':
                overrides['n_iter'] = 1
            if name == 'LinearDiscriminantAnalysis':
                overrides['n_components'] = 1
            if name == 'QuadraticDiscriminantAnalysis':
                overrides['reg_param'] = .5
            if name == 'SparseRandomProjection':
                overrides['compute_inverse_components'] = True
            obj = factory(public, **overrides)
        return obj

    def data():
        m = model()
        classifier = getattr(m, '_estimator_type', None) == 'classifier' or name.endswith('Classifier') or name.endswith('NB')
        y = labels if classifier else target
        if name.startswith('MultiOutput'):
            y = np.column_stack([y, y])
        if name in ('LabelEncoder', 'LabelBinarizer'):
            return labels, None
        if name == 'MultiLabelBinarizer':
            return [(0,), (1,), (0,1), (1,), (0,), (1,), (0,1), (0,)], None
        if name == 'IsotonicRegression':
            return X[:,0], target
        if name.startswith(('RNN', 'LSTM', 'GRU')):
            return X.reshape(8,2,2), y
        if name in ('CCA', 'PLSCanonical', 'PLSRegression'):
            return X, np.column_stack([target, target*.5])
        if name in ('Louvain',):
            return np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],np.float32), None
        if name in ('GridSearchCV', 'RandomizedSearchCV'):
            return X, labels
        if name in ('GradientBoosting','ExperimentalTwoLevelFeatureFreq','RFE'):return X,labels
        return X, y

    def fit():
        nonlocal fitted
        m = model()
        if not fitted:
            x, y = data()
            if name=='OrderedRMSE':
                m.fit(x,y,permutation=np.arange(len(x),dtype=np.int64))
            else:
                m.fit(x) if y is None else m.fit(x, y)
            fitted = True
        return m

    def consumed(m):
        x, _ = data()
        if hasattr(m, 'predict'):
            return m.predict(x)
        if hasattr(m, 'transform'):
            return m.transform(x)
        if hasattr(m, 'labels_'):
            return m.labels_
        if hasattr(m, 'kneighbors'):
            return m.kneighbors(x)
        raise ValueError('fixture needs explicit fitted output: '+public)

    def check(method):
        nonlocal fitted
        m = model()
        x, y = data()
        if method in ('fit_transform', 'fit_predict'):
            operation = getattr(m, method)
            result = operation(x) if y is None else operation(x,y)
            fitted = True
            return result
        if method in ('fit', 'partial_fit'):
            operation = getattr(m, method)
            result = operation(x) if y is None else operation(x,y)
            assert result is m
            fitted = True
            return consumed(m)
        m = fit()
        if name in ('OrderedRMSE','ExperimentalTwoLevelFeatureFreq') and method in ('predict_classes','predict_proba'):
            try:getattr(m,method)(x)
            except ValueError as error:
                assert 'classification loss' in str(error) or "fitted with 'RMSE'" in str(error)
                return {'classification_only_refused':True}
            raise AssertionError('RMSE admitted classification-only prediction')
        if method == 'score':
            if name in ('GaussianMixture','BayesianGaussianMixture','KernelDensity','FactorAnalysis'):
                return m.score(x)
            return m.score(x,y)
        if method == 'inverse_transform':
            return m.inverse_transform(m.transform(x))
        if method in ('aic','bic','predict_proba','predict_joint_log_proba','predict_log_proba',
                      'decision_function','apply','predict_classes','score_samples','transform',
                      'predict_with_leaves','latent_mean_and_variance'):
            return getattr(m,method)(x)
        if method == 'staged_predict':
            return list(m.staged_predict(x))
        if method == '__getstate__':
            return m.__getstate__()
        if method == '__setstate__':
            state=m.__getstate__()
            restored=object.__new__(type(m))
            restored.__setstate__(state)
            before,after=consumed(m),consumed(restored)
            from ._verify_api import encode
            assert encode(before)==encode(after)
            return after
        if method.endswith(':get'):
            value=getattr(m,method[:-4])
            if method=='estimators_:get':
                return [consumed(e) for e in value]
            return value
        if method=='estimators_:set':
            value=m.estimators_
            m.estimators_=value
            assert len(m.estimators_)==len(value)
            return consumed(m)
        if method in ('n_candidate_distances_','used_query_tile_'):
            m.kneighbors(x)
            if method=='n_candidate_distances_' and name!='NearestNeighbors':
                try:getattr(m,method)
                except AttributeError:
                    return {'availability':'RBC counter unavailable for brute classifier/regressor','numerical_execution':False}
                raise AssertionError('counter became available; add its numerical fixture')
            value=getattr(m,method)
            return value() if callable(value) else value
        if method=='log_marginal_likelihood':
            return m.log_marginal_likelihood()
        return getattr(m,method)()

    # Fit-returning methods execute once and provide state for read-only doors.
    order=sorted(set(methods)&METHODS,key=lambda m:(m not in ('fit_transform','fit_predict','fit','partial_fit'),m))
    for method in order:
        from . import _verify_api_availability as availability
        if public+'.'+method in availability.ABSENT | availability.CONDITIONAL:
            availability.check(recorder,public,method)
        else:
            recorder.check(public+'.'+method,lambda method=method:check(method))
