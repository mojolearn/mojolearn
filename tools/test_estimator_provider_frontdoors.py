"""Enforce one-call frontdoors; these source checks do not qualify runtime paths."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_selected_frontdoors_delegate_binding_admission_after_native_entry():
    expected = {
        'svm.py': {'LinearSVC': {'fit','predict','decision_function','score'},
                   'LinearSVR': {'fit','predict','score'}},
        '_svm_impl.py': {'SVC': {'fit','predict','predict_proba','predict_log_proba','decision_function','score'},
                        'SVR': {'fit','predict','score'}},
        '_expansion_cluster.py': {
            '_CentersMixin': {'predict', 'transform', 'fit_predict', 'fit_transform', 'score'},
            'MiniBatchKMeans': {'fit', 'partial_fit'},
            'BisectingKMeans': {'fit', 'predict'},
            'MeanShift': {'fit', 'predict', 'fit_predict'},
            'OPTICS': {'fit', 'fit_predict'},
            'AffinityPropagation': {'fit', 'predict', 'fit_predict'},
            'BayesianGaussianMixture': {'fit', 'fit_predict', 'predict', 'predict_proba', 'score_samples', 'score'},
        },
        'cluster.py': {'KMeans': {'fit','predict','transform','fit_predict','fit_transform'}},
    }
    checked = set()
    for filename, classes in expected.items():
        module = ast.parse((ROOT / 'python/mojolearn' / filename).read_text())
        for cls in (n for n in module.body if isinstance(n, ast.ClassDef) and n.name in classes):
            for method in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in classes[cls.name]):
                calls = [n for statement in method.body for n in ast.walk(statement) if isinstance(n, ast.Call)]
                assert len(calls) == 1, (filename, cls.name, method.name)
                call = calls[0]
                assert isinstance(call.func, ast.Name) and call.func.id == '_NATIVE'
                assert isinstance(call.args[2].elts[0], ast.Constant) and call.args[2].elts[0].value is None
                checked.add((filename, cls.name, method.name))
    assert len(checked) == sum(len(methods) for classes in expected.values() for methods in classes.values())

def test_one_shared_admission_helper_keeps_explicit_internal_arguments():
    source = (ROOT / 'providers/python/estimator_binding_pyapi.mojo').read_text()
    assert 'if len(args) == 0 or args[0] is not None:' in source
    assert 'estimator_binding(model, name, context)' in source
    family = (ROOT / 'bindings/family_native.mojo').read_text()
    for name in ('_mojolearn_estimators', '_mojolearn_svm', '_mojolearn'):
        assert f'estimator_arguments(model, "{name}", args, context)' in family


def test_import_connection_unwraps_host_proxy_without_attribute_callback(monkeypatch):
    import importlib.util
    import sys
    import types
    package = types.ModuleType('_provider_fixture'); package.__path__=[]
    backend = types.ModuleType('_provider_fixture._backend')
    class ModeSet:
        pass
    class HostBinding(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError('host facade hook must never be invoked')
    class NoGpuBinding(types.ModuleType):
        pass
    facade=HostBinding('fixture._mojolearn_x_neighbors')
    facade.__dict__['_HostBinding__basename']='_mojolearn_x_neighbors_host'
    real=types.ModuleType('fixture._host._mojolearn_x_neighbors_host')
    backend._ModeSet=ModeSet;backend._HostBinding=HostBinding;backend._NoGpuBinding=NoGpuBinding
    backend._CPU_ONLY="explicit CPU test fixture"
    backend._offers=lambda name,mode: mode=='identical'
    backend._vendor_fn=lambda name:'x_neighbors_vendor'
    backend._host_prefix=lambda basename:'x_neighbors_host'
    loaded=[]
    def load(basename):
        loaded.append(basename);return real
    backend.load_host_module=load
    def binding(name,mode):
        if mode!='identical':raise ValueError('host bindings are identical only')
        return facade
    backend.binding=binding
    monkeypatch.setitem(sys.modules,package.__name__,package)
    monkeypatch.setitem(sys.modules,backend.__name__,backend)
    spec=importlib.util.spec_from_file_location('_provider_fixture.bridge',ROOT/'python/mojolearn/_native_family_bridge.py')
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    context={};bridge.shared_context=lambda:context
    bridge.connect_bindings('_mojolearn_x_neighbors')
    modes=context['family_binding_connections']['_mojolearn_x_neighbors']
    assert modes['identical']==(real,real.__name__,'x_neighbors_host_numeric_mode')
    assert modes['fast'][0].startswith('ValueError:')
    assert loaded==['_mojolearn_x_neighbors_host']
    assert context['family_binding_offerings']['_mojolearn_x_neighbors']=={'identical':True,'fast':False}


def test_native_admission_preserves_mode_check_and_live_host_replacement():
    source=(ROOT/'providers/python/estimator_binding_pyapi.mojo').read_text()
    assert 'context["family_binding_offerings"][name][requested]' in source
    assert 'py_get(native_loaded_module("sys").modules, record[1])' in source
    assert 'neural_descriptor_entry(module, reader, False)' in source
    assert 'if got != expected:' in source
    assert 'native host binding must be an exact extension module' in source
    assert '.load_host_module(' not in source and '._HostBinding' not in source


def test_cluster_native_boundary_refuses_initializer_callbacks_before_array_conversion():
    source = (ROOT / 'bindings/cluster_fit_a_native.mojo').read_text()
    assert 'def _callable_init(' not in source
    helper = source.split('def _init_array(', 1)[1].split('def _flag(', 1)[0]
    assert helper.index('PyCallable_Check') < helper.index('_f32(init,')
    assert 'pass explicit initial centers' in helper
    seed = source.split('def _seed(', 1)[1].split('def _weights(', 1)[0]
    assert 'py_type(random_state) is not b.int' in seed


def test_cluster_classes_disable_python_decorators_and_keep_numeric_mode_signature():
    module = ast.parse((ROOT / 'python/mojolearn/_expansion_cluster.py').read_text())
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    flags = [node for node in classes['_XCluster'].body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == '_mojolearn_native_public'
                     for target in node.targets)]
    assert len(flags) == 1 and flags[0].value.value is True
    for name in ('MiniBatchKMeans', 'BisectingKMeans', 'MeanShift', 'OPTICS',
                 'AffinityPropagation', 'BayesianGaussianMixture'):
        initializer = next(node for node in classes[name].body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        assert 'numeric_mode' in [arg.arg for arg in initializer.args.kwonlyargs], name
        for method in (node for node in classes[name].body if isinstance(node, ast.FunctionDef)
                       and node.name in ('fit', 'partial_fit', 'fit_predict', 'fit_transform')):
            call = next(node for node in ast.walk(method) if isinstance(node, ast.Call))
            assert isinstance(call.args[2].elts[-1], ast.Name) and call.args[2].elts[-1].id == 'y'


def test_cluster_lifecycle_keeps_required_checks_and_result_wrapping_native():
    source = (ROOT / 'providers/python/cluster_public_lifecycle_pyapi.mojo').read_text()
    for required in ('classical_check_x(', 'classical_require_fitted(', 'estimator_feature_check(',
                     'estimator_fit_begin(', 'estimator_fit_end(', 'estimator_fit_complete(',
                     'native_array_ndarray_view(', 'host_inference_scalar('):
        assert required in source
    assert 'finally:\n        if fitting:\n            estimator_fit_end(model, depth)' in source
    assert 'Python init callbacks are not accepted' in source


def test_cluster_context_lookup_cannot_enter_module_getattr():
    source = (ROOT / 'providers/python/cluster_public_lifecycle_pyapi.mojo').read_text()
    assert 'native_loaded_module("mojolearn._native_public_attributes")._CONTEXT' not in source
    assert 'py_get(module_fields, "_CONTEXT")' in source
    assert 'py_type(attributes) is not b.dict' in source


def test_cluster_learned_scalars_have_numeric_dtype_admission():
    source = (ROOT / 'providers/python/cluster_public_lifecycle_pyapi.mojo').read_text()
    learned = source.split('var learned =', 1)[1].split('return fields', 1)[0]
    for name in ('n_features_in_', '_partial_batch', '_partial_init', '_partial_state', 'n_steps_', 'lower_bound_'):
        assert name in learned
    assert 'host_inference_scalar(value, context)' in learned
    assert '_scalar_state(value, context)' in learned
    assert '_scalar_state(value[0], context)' in learned
