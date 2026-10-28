"""Source admission/ownership checks; root separately executes public probes."""
import ast
import re
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return (ROOT / path).read_text()

def body(source, name):
    return re.search(rf'^def {name}\(.*?(?=^def |\Z)', source, re.M | re.S).group()

def test_native_owner_has_no_provider_or_python_dependency():
    _, errors = audit(ROOT, ('kernel_methods.native_descriptor_dispatch', 'kernel_methods.descriptor_schema'))
    assert not errors, errors
    source = read('kernel_methods/native_descriptor_dispatch.mojo')
    assert 'PythonObject' not in source and 'py=' not in source
    assert 'native_descriptor_report(problem, context, sink)' in source

def test_public_frontdoors_only_enter_native_owner():
    module = ast.parse(read('python/mojolearn/kernel_methods.py'))
    expected = {'KernelRidge': {'fit', 'predict'}, 'Nystroem': {'fit', 'transform', 'fit_transform'},
                'RBFSampler': {'fit', 'transform', 'fit_transform'}}
    for cls in (node for node in module.body if isinstance(node, ast.ClassDef) and node.name in expected):
        for method in (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in expected[cls.name]):
            calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
            assert len(calls) == 1, (cls.name, method.name, [ast.unparse(n) for n in calls])
            assert isinstance(calls[0].func, ast.Name) and calls[0].func.id == '_NATIVE'

def test_shared_binding_admission_uses_live_registry_without_load_callbacks():
    source = read('providers/python/estimator_binding_pyapi.mojo')
    assert 'backend["_SETS"]' in source and 'fields["_modules"]' in source
    assert 'context["family_mode_set_fields"]' in source
    assert 'py_type(module) is not py_type(bi)' in source
    # No runtime calls to Python loader helpers or arbitrary mode-set hooks.
    for text in ('py_call(selected.', 'py_call(backend', '._bind(', '.load_set('):
        assert text not in source

def test_computation_transport_reuses_common_provider_and_resource():
    source = read('providers/python/kernel_method_descriptor_pyapi.mojo')
    assert 'struct ' not in source
    assert 'neural_descriptor_reuse(previous, factory' in source
    assert 'neural_descriptor_execute(native, operation' in source
    source = read('bindings/kernel_methods_native.mojo')
    assert 'neural_descriptor_adopt(native, Int(py=handle), 5, [])' in source
    assert 'neural_descriptor_resource_close(held[3], [])' in source
    assert 'held[4] is native' in source
    assert 'state.pop, "_native_kernel_method_provider"' in source

def test_compound_transform_reuses_admitted_x_and_provider():
    source = read('bindings/kernel_methods_native.mojo')
    for name, operation in [('_nys_fit', 'nystroem_transform'), ('_rbf_fit', 'rbf_sampler_transform')]:
        part = body(source, name)
        assert part.count('family_f32_snapshot(args[1]') == 1
        assert f'kernel_method_execute(native, "{operation}"' in part

def test_native_transport_preserves_existing_numerical_owners_and_route_defines():
    native = read('kernel_methods/native_descriptor_dispatch.mojo')
    original = read('bindings/_mojolearn_kernel_methods.mojo')
    for name in ('kernel_ridge_fit', 'kernel_ridge_predict', 'kernel_ridge_fit_ptr', 'kernel_ridge_predict_ptr',
                 'kernel_ridge_resident_prepare', 'kernel_ridge_resident_release', 'kernel_ridge_predict_resident',
                 'nystroem_fit', 'nystroem_fit_ptr', 'nystroem_fit_transform_ptr', 'nystroem_transform',
                 'rbf_sampler_fit', 'rbf_sampler_transform'):
        a, b = body(original, name + '_binding'), body(native, 'km_native_' + name)
        numerical = r'\b(?:_kernel_ridge_\w+_run|_nystroem_\w+_run|_rbf_sampler_\w+_run|kernel_ridge_\w+_into|kernel_ridge_resident_\w+|kernel_ridge_predict_resident|km_nystroem_\w+_run|nystroem_transform_ptr_into|rbf_sampler_transform_ptr_into)\('
        assert [v for v in re.findall(numerical, a) if not v.endswith("_binding(")] == re.findall(numerical, b), name
    assert 'Int(KRR_IDN_PTR_IN)' in native and 'Int(NYS_IDN_FIT_PTR_IN)' in native
    assert native.count('comptime if KM_FAST_PTR_IN:') == 2

def test_private_span_bounds_and_scalar_schemas_are_checked_before_dispatch():
    source = body(read('kernel_methods/native_descriptor_dispatch.mojo'), 'km_native_v1')
    assert source.index('na > 8') < source.index('native_descriptor_words(aa, na)')
    assert 'nf != np' in source
    assert 'operation == 6 and na != 3' in source
    assert 'operation == 5 and (na != 1 or np != 0)' in source
    source = read('providers/python/kernel_method_descriptor_pyapi.mojo')
    assert source.index('len(params) != schema[0]') < source.index('params[i]')


def test_common_array_metadata_snapshot_retains_owner_and_native_shape():
    source = read('providers/python/array_metadata_pyapi.mojo')
    assert 'var keep: PythonObject' in source and 'var source: CastIn' in source
    for field in ('value.shape', 'value.dtype', 'value.itemsize', 'value.strides'):
        assert source.count(field) == 1, field
    conversion = body(read('bindings/family_common.mojo'), 'family_f32_snapshot')
    assert conversion.count('array_shape_once(view, context)') == 1
    assert 'family_cast_rows(src.addr, src.s0, src.s1, src.rows, src.cols, src.itemsize, src.kind' in conversion
    assert 'snapshot.replace_f32(result, address)' in conversion
    source = read('bindings/kernel_methods_native.mojo')
    for name in ('_krr_fit', '_krr_predict', '_nys_fit', '_nys_transform', '_rbf_fit', '_rbf_transform'):
        part = body(source, name)
        assert part.count('family_f32_snapshot(args[1]') == 1, name
        for text in ('x.shape', 'xq.shape', 'x.size', 'xq.size', 'classical_address(x)', 'classical_address(xq)'):
            assert text not in part, (name, text)
    assert 'family_owned_snapshot(x_meta, context)' in source
    assert '_km_scale_gamma(extension, x_meta, d, context, native)' in source
