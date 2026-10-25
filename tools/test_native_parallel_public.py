# SPDX-License-Identifier: Apache-2.0
"""Source and ABI contracts for public drivers; no package/native import."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'python/mojolearn'


def _functions(module):
    tree = ast.parse((PACKAGE / (module + '.py')).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, node
        elif isinstance(node, ast.ClassDef) and node.name == 'ParallelForestPredictor':
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    yield node.name + '.' + method.name, method


@pytest.fixture
def connected(monkeypatch):
    package = types.ModuleType('_parallel_fixture')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module_names = ('linear_model', 'decomposition', '_solver_impl', 'cluster', '_svm_impl',
        '_gp_impl', 'density', 'mixture', 'hdbscan', '_cholesky_impl', 'kernel_methods',
        '_hierarchy_impl', '_spectral_impl', '_umap_impl', '_expansion_cluster', 'ensemble',
        '_gbdt_adapters', '_iforest_impl', 'randomforest', 'extratrees')
    classes = {}
    for name in module_names:
        module = types.ModuleType(package.__name__ + '.' + name)
        tree = ast.parse((PACKAGE / (name + '.py')).read_text())
        sources = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        literal_globals = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        literal_globals[target.id] = value
                        setattr(module, target.id, value)
        for cname, node in sources.items():
            cls = type(cname, (), {})
            attrs = {}
            def read_attributes(current):
                for base in current.bases:
                    if isinstance(base, ast.Name) and base.id in sources:
                        read_attributes(sources[base.id])
                for attr in current.body:
                    if isinstance(attr, ast.Assign):
                        try:
                            value = literal_globals[attr.value.id] if isinstance(attr.value, ast.Name) else ast.literal_eval(attr.value)
                        except (KeyError, ValueError, TypeError):
                            continue
                        for target in attr.targets:
                            if isinstance(target, ast.Name):
                                attrs[target.id] = value
            read_attributes(node)
            for key, value in attrs.items():
                setattr(cls, key, value)
            setattr(module, cname, cls)
            classes[(name, cname)] = cls
        # Distinct owner contexts catch accidentally using another module's ABI.
        for entry in ('_LR_CALL', '_native_classical', '_PCA_ENTRY', '_TSVD_ENTRY', '_CD_CALL', '_TREES'):
            setattr(module, entry, len)
        for owner in ('_LR_CONTEXT', '_CLASSICAL', '_PCA_CONTEXT', '_TSVD_CONTEXT', '_CD_CONTEXT', '_TREES_CONTEXT'):
            setattr(module, owner, {'module': name, 'owner': owner})
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(package, name, module)
    backend = types.ModuleType(package.__name__ + '._backend')
    backend._CPU_ONLY = None
    backend._DEFAULT_MODE = 'identical'
    backend.default_mode = lambda: backend._DEFAULT_MODE
    backend._HostBinding = type('HostBinding', (types.ModuleType,), {})
    def binding(name, mode):
        module = types.ModuleType(name + ':' + mode)
        module.compiled_entry = len
        module.unsafe_python_helper = lambda: None
        return module
    backend.binding = binding
    family = types.ModuleType(package.__name__ + '._native_family_bridge')
    family.connect = lambda: (len, {})
    shard = types.ModuleType(package.__name__ + '._native_parallel_shard')
    shard._native_parallel_shard = len
    shard.register = lambda **kwargs: None
    for module in (backend, family, shard):
        monkeypatch.setitem(sys.modules, module.__name__, module)
        setattr(package, module.__name__.rsplit('.', 1)[-1], module)
    spec = importlib.util.spec_from_file_location(package.__name__ + '.public', PACKAGE / '_native_parallel_public.py')
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    context = {}
    connector.connect(context)
    return context['parallel_public'], classes


def test_every_public_operation_including_lifecycle_has_one_raw_native_call(connected):
    meta, _ = connected
    found = set()
    total = 0
    for module in ('parallel_classical', 'parallel_graph', 'parallel_ensemble'):
        for name, node in _functions(module):
            body = node.body[1:] if ast.get_docstring(node) else node.body
            assert len(body) == 1 and isinstance(body[0], ast.Return), (module, name)
            call = body[0].value
            assert isinstance(call, ast.Call), (module, name)
            assert len([n for n in ast.walk(call) if isinstance(n, ast.Call)]) == 1, (module, name)
            if not isinstance(call.func, ast.Name) or call.func.id != '_FAMILY':
                assert name in ('fit_arima', 'fit_exponential_smoothing', 'fit_forest')
                continue
            total += 1
            assert ast.literal_eval(call.args[1]) == 'family_par_public'
            operation = ast.literal_eval(call.args[2].elts[0])
            raw = call.args[2].elts[1]
            assert isinstance(raw, ast.Tuple)
            assert all(isinstance(arg, ast.Name) for arg in raw.elts)
            # Execute only the one-call shell with unique raw sentinels: no
            # package import, descriptors, preparation, or native execution.
            seen = []
            namespace = {'_FAMILY': lambda *args: seen.append(args) or 'done', '_FAMILY_CONTEXT': object()}
            standalone = ast.FunctionDef(name=node.name, args=node.args, body=body,
                decorator_list=[], returns=None, type_comment=None, type_params=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[standalone], type_ignores=[])), '<shell>', 'exec'), namespace)
            fn = namespace[node.name]
            signature = inspect.signature(fn)
            args, kwargs = [], {}
            for parameter in signature.parameters.values():
                if parameter.kind == parameter.VAR_POSITIONAL:
                    continue
                if parameter.kind == parameter.KEYWORD_ONLY:
                    kwargs[parameter.name] = object()
                else:
                    args.append(object())
            assert fn(*args, **kwargs) == 'done'
            assert len(seen) == 1
            if operation.startswith('predictor_'):
                continue
            found.add(operation)
            for plan in meta['plans'][operation].values():
                assert raw.elts[plan['device_arg']].id == 'devices'
                if plan['numeric_mode_arg'] is not None:
                    assert raw.elts[plan['numeric_mode_arg']].id == 'numeric_mode'
    assert found == set(meta['plans'])
    assert total == 33  # 27 functions plus constructor, queries, close and context manager.


def test_native_owner_argument_recipes_preserve_nondefault_supported_inputs(connected):
    meta, classes = connected
    # Independent public-owner contracts, with all optional arguments distinct.
    expected = {
        ('fit_logistic','linear_model','LogisticRegression'): ('fit',(0,1,None)),
        ('fit_gram_estimator','linear_model','Ridge'): ('ridge_fit',(0,1,3)),
        ('fit_gram_estimator','decomposition','PCA'): ('fit',(0,1)),
        ('fit_gram_estimator','decomposition','TruncatedSVD'): ('fit',(0,1)),
        ('fit_coordinate_descent','_solver_impl','ElasticNet'): ('cd_fit',(0,1,None)),
        ('fit_boosting','ensemble','GradientBoosting'): ('trees_gbdt_fit',(0,1,3,4,None,None,None,None)),
        ('fit_boosting','_gbdt_adapters','GradientBoostingRegressor'): ('trees_gbdt_adapter_regressor_fit',(0,1,3,4)),
        ('fit_isolation_forest','_iforest_impl','IsolationForest'): ('trees_iforest_fit',(0,None)),
        ('fit_ordered_rmse','ensemble','OrderedRMSE'): ('trees_gbdt_ordered_fit',(0,1,2,4)),
        ('fit_feature_freq','ensemble','ExperimentalTwoLevelFeatureFreq'): ('trees_gbdt_feature_freq_fit',(0,1,3)),
    }
    for (operation,module,cls), (owner,indices) in expected.items():
        plan = meta['plans'][operation][classes[module,cls]]
        assert plan['operation'] == owner
        assert plan['recipe'] == tuple(('constant', None) if i is None else ('arg',i) for i in indices)
        assert plan['owner']['module'] == module
    for operation, before, after in (('bootstrap',9,(10,)),('permutation_test',7,(8,)),('monte_carlo_integrate',6,())):
        plan = meta['plans'][operation][None]
        assert plan['recipe'] == (('binding','_mojolearn_resample','identical'),) + tuple(('arg',i) for i in range(before)) + (('constant','identical'),) + tuple(('arg',i) for i in after)
    for cls in ('KernelRidge','Nystroem'):
        plan = meta['plans']['fit_kernel_method'][classes['kernel_methods',cls]]
        assert plan['route'] == 'direct'
        assert plan['recipe'] == (('binding','_mojolearn_kernel_methods','model'),('arg',0),('arg',1),('arg',2))
    plan = meta['plans']['transform_rbf_sampler'][classes['kernel_methods','RBFSampler']]
    assert plan['route'] == 'shard'
    assert plan['recipe'] == (('model',),('arg',0),('arg',1),('arg',2),('binding','_mojolearn_kernel_methods','model'))


def test_binding_namespaces_cannot_invoke_module_hooks_or_python_helpers(connected):
    meta, _ = connected
    for modes in meta['modules'].values():
        for module in modes.values():
            assert type(module) is types.SimpleNamespace
            assert set(vars(module)) == {'__name__','compiled_entry'}
            assert type(module.compiled_entry) is types.BuiltinFunctionType
    assert len(meta['plans']) == 27


def test_native_boundary_admits_before_hooks_and_scopes_predictor_lifetime():
    source = (ROOT / 'bindings/parallel_public_native.mojo').read_text()
    for unsafe in ('Python.import_module', 'copy.copy', 'threading', '._finalizer', 'model.fit(', 'model.predict(', '._bind(', 'public_method_result'):
        assert unsafe not in source
    assert source.index('py_type(value) is not b.tuple') < source.index('py_call(b.tuple, value)')
    assert 'py_type(cls) is not b.type' in source
    assert 'parallel estimator methods or storage hooks changed' in source
    assert 'if py_type(lock) is not meta["lock_type"]' in source
    assert 'finally:\n        _ = py_call(lock.release)' in source
    assert 'py_setitem(saved, "_resident_holder", Python.dict())' in source
    assert 'parallel_public_result(result,' in source
    result = (ROOT / 'bindings/parallel_public_result.mojo').read_text()
    assert 'String(value.dtype.kind) == "f"' in result
    assert '_outside(context, 1)' in result
