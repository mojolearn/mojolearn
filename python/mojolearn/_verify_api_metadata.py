"""Tiny public constructor/parameter/attribute contracts, no fitting."""
__mojolearn_user_boundary__ = True

import inspect
import math

from ._verify_api import resolve

PARAMETER_METHODS = ('get_params', 'set_params', '__getattribute__')


def factory(public, **overrides):
    import numpy as np
    from . import Ridge, LogisticRegression, DecisionTreeRegressor
    cls = resolve(public)
    name = public.rsplit('.', 1)[-1]
    sig = inspect.signature(cls)
    small = dict(numeric_mode='identical', random_state=7, max_iter=2, n_iter=2, epochs=1,
                 n_estimators=2, max_depth=2, n_components=2, n_neighbors=2,
                 n_clusters=2, n_bins=2, cv=2, n_init=1, batch_size=4,
                 hidden_layer_sizes=(2,), hidden_size=2, max_epochs=1, n_inducing=2,
                 transform_max_iter=2, n_splits=2, n_repeats=1,
                 n_lists=2, n_probes=1, num_embeddings=4, embedding_dim=2,
                 n_groups=1, p=1, n_heads=1)
    kwargs = {k: v for k, v in small.items() if k in sig.parameters}
    if name != 'LeavePOut':
        kwargs.pop('p', None)
    for key, value in dict(n_epochs=2, kmeans_n_iters=2, pq_kmeans_n_iters=2,
                           pq_bits=1, pq_dim=2, k=2, sparse_output=False).items():
        if key in sig.parameters:
            kwargs[key]=value
    if 'shuffle' in sig.parameters and sig.parameters['shuffle'].default is False:
        kwargs['shuffle'] = True
    X = np.array([[.125, .25], [.25, .5], [.5, .125], [.75, .5]], np.float32)
    required = dict(params=[np.array([.25, -.5], np.float32)],
                    endog=np.arange(8, dtype=np.float32) / 8, data=X,
                    estimator=LogisticRegression(max_iter=2) if 'Classifier' in name else Ridge(),
                    model=DecisionTreeRegressor(max_depth=1) if name == 'TreeExplainer' else Ridge(),
                    estimators=[('tiny', LogisticRegression(max_iter=2) if 'Classifier' in name else Ridge())],
                    param_grid={'alpha': [.25]}, param_distributions={'alpha': [.25]},
                    test_fold=np.array([0, 1, 0, 1], np.int64))
    required['dictionary'] = np.eye(4, dtype=np.float32)
    if name == 'Embedding':
        kwargs['weight'] = np.arange(8, dtype=np.float32).reshape(4, 2) / 16
    if name == 'ExperimentalTwoLevelFeatureFreq':
        required['sources'] = [0, 1]
    if name == 'STL':
        kwargs['period'] = 2
    if name in ('KernelExplainer', 'PermutationExplainer', 'TreeExplainer'):
        required['model'].fit(X, np.array([.25, .5, .75, 1], np.float32))
        kwargs['data'] = X
    if name.startswith(('Mamba', 'TransformerBlock')):
        required['weights'] = block_weights(name)
    if name == 'VAR':
        required['endog'] = X
    for key, parameter in sig.parameters.items():
        if parameter.default is parameter.empty and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) and key not in kwargs:
            if key not in required:
                raise ValueError('fixture requires explicit constructor value: ' + public + '.' + key)
            kwargs[key] = required[key]
    kwargs.update(overrides)
    return cls(**kwargs)


def parameter_checks(recorder, public):
    prefix = public + '.'

    def get_params():
        obj = factory(public)
        shallow = obj.get_params(deep=False)
        assert isinstance(shallow, dict)
        names = [k for k, p in inspect.signature(type(obj)).parameters.items()
                 if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
        if 'numeric_mode' in shallow and 'numeric_mode' not in names:
            names.append('numeric_mode')
        assert set(shallow) == set(names), (sorted(shallow), sorted(names))
        for key, value in shallow.items():
            if hasattr(obj, key):
                assert getattr(obj, key) is value, key
        deep = obj.get_params(deep=True)
        assert all(k in deep and deep[k] is v for k, v in shallow.items())
        return shallow

    def attributes():
        obj = factory(public)
        # Real public field access and a missing-field refusal exercise the
        # installed descriptor route on every concrete exposed class.
        params = ({'numeric_mode': obj.numeric_mode} if public.rsplit('.',1)[-1].startswith(('Mamba','TransformerBlock'))
                  else obj.get_params(deep=False))
        if not params:
            # Parameterless estimators still expose their real class descriptor.
            assert obj.__getattribute__('__class__') is type(obj)
            return {'class': type(obj).__name__}
        key = next(k for k in params if hasattr(obj, k))
        value = obj.__getattribute__(key)
        assert value is params[key]
        try:
            obj.__getattribute__('missing_verifier_attribute_')
        except AttributeError:
            pass
        else:
            raise AssertionError('missing attribute must raise AttributeError')
        return {key: value, 'missing_attribute_refused': True}

    def set_params():
        obj = factory(public)
        params = obj.get_params(deep=False)
        assert obj.set_params() is obj
        choices = [(k, v) for k, v in params.items() if type(v) in (bool, int, float) and k != 'random_state'
                   and (type(v) is not float or math.isfinite(v))]
        if public.rsplit('.',1)[-1].startswith(('Mamba','TransformerBlock')):
            choices = [('weights', params['weights'])]
        if public.endswith('.Embedding'):
            choices = [('padding_idx', params['padding_idx'])]
        if public.endswith('.UMAP'):
            choices = [('n_neighbors', params['n_neighbors'])]
        if not choices:
            choices = [(k, v) for k, v in params.items() if type(v) in (bool, int, float)]
        if not choices:
            if 'weights' in params:
                choices = [('weights', {k:v.copy() for k,v in params['weights'].items()})]
            elif 'jitter' in params:
                choices = [('jitter', .125)]
            elif 'alias' in params:
                choices = [('alias', 'verifier')]
            elif 'shrink_threshold' in params:
                choices = [('shrink_threshold', .125)]
            elif 'endog' in params:
                choices = [('endog', params['endog'] + .125)]
            elif 'estimator' in params:
                choices = [('estimator', factory('mojolearn.Ridge', alpha=.75))]
            elif 'data' in params:
                choices = [('data', params['data'].copy())]
            elif not params:
                try:
                    obj.set_params(unknown_verifier_parameter=1)
                except ValueError:
                    return {'empty_parameters': True, 'unknown_refused': True}
                raise AssertionError('parameterless estimator accepted unknown parameter')
            else:
                raise ValueError('fixture needs explicit parameter mutation for ' + public)
            key, value = choices[0]
            before = params[key]
        else:
            key, before = choices[0]
            value = ({k:v+.125 for k,v in before.items()} if key=='weights' else 1) if key in ('weights','padding_idx') else (not before if type(before) is bool else before + (1 if type(before) is int else .125))
        assert obj.set_params(**{key: value}) is obj
        after = obj.get_params(deep=False)
        assert after[key] is value
        if hasattr(obj, key):
            assert getattr(obj, key) is value
        try:
            if '.model_selection.' in public:
                obj.set_params(**{'fit': 1})
            else:
                obj.set_params(unknown_verifier_parameter=1)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('invalid/protected parameter must be refused')
        result = {'parameter': key, 'before': before, 'after': after[key], 'invalid_refused': True}
        if key == 'weights':
            import numpy as np
            from ._verify_api import encode
            x = np.arange(64, dtype=np.float32).reshape(1, 2, 32) / 128 - .25
            actual = obj(x)
            fresh = type(obj)(**after)(x)
            assert encode(actual) == encode(fresh), 'set_params did not rebuild executable weights'
            result['forward'] = actual
        return result

    for method, check in [('get_params', get_params), ('set_params', set_params), ('__getattribute__', attributes)]:
        recorder.check(prefix + method, check, kind='metadata/state contract')


def block_weights(name):
    import numpy as np
    dm, di, nh = 32, 64, 1
    if name.startswith('Mamba1'):
        shapes = {'norm.weight': (dm,), 'in_proj.weight': (2*di, dm), 'conv1d.weight': (di,1,4),
                  'conv1d.bias': (di,), 'x_proj.weight': (34,di), 'dt_proj.weight': (di,2),
                  'dt_proj.bias': (di,), 'A_log': (di,16), 'D': (di,), 'out_proj.weight': (dm,di)}
    elif name.startswith('Mamba2'):
        shapes = {'block_norm.weight': (dm,), 'in_proj.weight': (2*di+256+nh,dm),
                  'conv1d.weight': (di+256,1,4), 'conv1d.bias': (di+256,), 'dt_bias': (nh,),
                  'A_log': (nh,), 'D': (nh,), 'norm.weight': (di,), 'out_proj.weight': (dm,di)}
    elif name.startswith('Mamba3'):
        shapes = {'block_norm.weight': (dm,), 'in_proj.weight': (2*di+256+3*nh+32,dm),
                  'dt_bias': (nh,), 'B_norm.weight': (128,), 'C_norm.weight': (128,),
                  'B_bias': (nh,128), 'C_bias': (nh,128), 'D': (nh,), 'out_proj.weight': (dm,di)}
    else:
        shapes = {'input_layernorm.weight': (dm,), 'post_attention_layernorm.weight': (dm,),
                  'q_proj.weight': (dm,dm), 'k_proj.weight': (dm,dm), 'v_proj.weight': (dm,dm),
                  'o_proj.weight': (dm,dm), 'gate_proj.weight': (di,dm), 'up_proj.weight': (di,dm),
                  'down_proj.weight': (dm,di)}
    # These blocks have fixed native state/head widths. Use one token later;
    # reducing supported weight widths below the native contract is invalid.
    return {key: (np.ones(shape, np.float32) if 'norm.weight' in key else
                  ((np.arange(np.prod(shape), dtype=np.float32).reshape(shape)+7*i)%11-5)/32)
            for i,(key,shape) in enumerate(sorted(shapes.items()))}
