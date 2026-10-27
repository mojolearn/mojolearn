"""Architectural/source regressions; no ABI or numerical-runtime qualification."""
import json
from check_generation_native_boundary import ROOT, audit


def read(name):
    return (ROOT / name).read_text()


def test_native_closure_includes_shared_numerical_owners_without_cpython():
    seen, failures = audit()
    assert {'training.samba_ops', 'mamba.impl.mamba_shell_impl', 'core.native_descriptor_call'} <= seen
    assert not failures, failures


def test_connection_and_hot_execution_remain_separate():
    provider = read('providers/python/neural_descriptor_pyapi.mojo')
    connect = provider.split('def neural_descriptor_connect(')[1].split('def neural_descriptor_cached')[0]
    assert 'PyCapsule_IsValid' in connect and 'PyCapsule_GetPointer' in connect
    assert 'NeuralDescriptorProvider(module, capsule, factory, name^, table[0])' in connect
    execute = provider.split('def neural_descriptor_execute(')[1].split('def neural_descriptor_info')[0]
    assert 'with GILReleased(Python())' in execute and '_ = provider' in execute
    assert 'py_call' not in execute and 'getattr' not in execute and 'getenv' not in execute
    for file in ('bindings/samba_inference_native.mojo', 'bindings/samba_training_native.mojo'):
        assert 'neural_descriptor_cached' not in read(file)
    for file in ('bindings/samba_state_native.mojo', 'bindings/causal_lm_construct.mojo'):
        assert 'neural_descriptor_cached' in read(file)


def test_both_doors_share_numerical_functions_and_original_float_casts():
    for file, fn in [('bindings/_mojolearn_training.mojo', 'native_primitive_operation'),
                     ('bindings/_mojolearn_mamba.mojo', 'mamba_state_operation')]:
        body = read(file)
        for op in range(6):
            assert f'{fn}({op},' in body
    native = read('training/native_primitive_dispatch.mojo')
    for family in ('embedding', 'rms_norm', 'linear'):
        for direction in ('forward', 'backward'):
            assert f'return samba_{family}_{direction}_host(' in native
    assert native.count('Float32(f[0])') == 2
    assert 'MojoNeuralTrainingContextIdentical' in native
    mamba = read('mamba/native_state_dispatch.mojo')
    assert '_mamba1_run(a, b, l, dm, step, position)' in mamba
    assert '_mamba2_run(a, b, l, dm, p[tail], Float32(f[0]), Float32(f[1]))' in mamba
    assert '_mamba3_run(a, b, l, dm, p[tail], p[tail + 1])' in mamba


def test_native_descriptors_are_checked_named_private_operations():
    abi = read('core/native_descriptor_abi.mojo')
    assert 'mojolearn.training.primitives.v1' in abi and 'mojolearn.mamba.state.v1' in abi
    for file in ('training/native_primitive_dispatch.mojo', 'mamba/native_state_dispatch.mojo'):
        body = read(file)
        assert 'if op < 0 or op >= 6:' in body
        assert 'if result == 0:' in body
        assert 'native_descriptor_report(problem, context, sink)' in body
    for file in ('providers/python/neural_descriptor_pyapi.mojo', 'core/native_descriptor_call.mojo'):
        body = read(file)
        for forbidden in ('PyCFunction_GetFunction', 'PyObject_Vectorcall', 'PyObject_Call'):
            assert forbidden not in body


def test_coverage_lists_actual_twenty_nine_plan_rows_and_pending_runtime():
    import bench_board
    actual = bench_board.plan_races('nvidia', ['identical'], families=['neural'])
    report = json.loads(read('experiments/neural_native_dispatch_20261027/coverage.json'))
    assert len(report['recipes']) == 29
    assert {r['id'] for r in report['recipes']} == {r['id'] for r in actual}
    assert all(r['runtime_validation'] == 'pending_local_mac' for r in report['recipes'])
    assert all(r['whole_recipe_python_free'] is False for r in report['recipes'])
    assert report['candidate_default'] == 'off'


def test_result_and_error_share_stable_caller_owned_storage():
    caller = read('core/native_descriptor_call.mojo')
    assert 'List[NativeCallOutput]()' in caller
    assert 'Int(MutPointer(to=output[0].result))' in caller
    assert 'Int(MutPointer(to=output[0].error))' in caller
    assert 'return output[0].result' in caller
    assert 'Array[Int, 1]' not in caller
