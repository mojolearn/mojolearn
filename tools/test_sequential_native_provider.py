"""Source-only boundary checks; public execution is qualified separately."""
import ast
from pathlib import Path

from check_neural_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]


def test_shared_model_execution_closure_cannot_reenter_a_provider():
    roots = ('autograd.model_execution', 'autograd.model_descriptor',
             'sequence.model_device_dispatch', 'sequence.model_host_dispatch')
    result = audit(roots=roots)
    assert result['passed'], result['failures']
    assert 'autograd.model' in result['source_modules']


def test_sequential_no_longer_dispatches_whole_model_through_cpython():
    text = (ROOT / 'bindings/sequential_execution.mojo').read_text()
    assert 'py_call_typed' not in text
    for entry in ('model_n_params', 'model_fit', 'model_fit_labels',
                  'model_predict', 'model_predict_labels'):
        assert f'entries["{entry}"]' not in text
    assert 'model_counts_native(native_program, native_table)' in text
    assert 'neural_descriptor_reuse(' in text
    assert 'String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)' in text
    for operation in ('MODEL_FIT', 'MODEL_FIT_LABELS', 'MODEL_PREDICT', 'MODEL_PREDICT_LABELS'):
        assert f'neural_descriptor_execute(provider, {operation},' in text


def test_legacy_packed_doors_and_native_dispatch_share_engine():
    packed = (ROOT / 'providers/python/autograd_pyapi.mojo').read_text()
    direct = (ROOT / 'autograd/model_descriptor.mojo').read_text()
    for name in ('model_fit_native', 'model_predict_native', 'model_class_count'):
        assert name in packed and name in direct
    for target in ('device', 'host'):
        native = (ROOT / f'sequence/model_{target}_dispatch.mojo').read_text()
        binding = (ROOT / ('bindings/_mojolearn_x_sequence' + ('_host' if target == 'host' else '') + '.mojo')).read_text()
        assert 'model_request_execute(' in native
        assert 'neural_descriptor_capsule(api[].name, api[].entries)' in binding
        assert '"model_native_api_v1"' in binding


def test_program_owners_survive_native_execution():
    source = (ROOT / 'providers/python/autograd_program_pyapi.mojo').read_text()
    replay = source.index('var r = program_execute_run[E, KIND](sid_i, plan, inputs^, profiled)')
    assert source.index('_ = arrays', replay) < source.index('host_arrays_py', replay)
    fit = source.index('var branches = program_fit_native[E, KIND](')
    out = source.index('var out = Python.list()', fit)
    for owner in ('arrays', 'targets', 'curve'):
        assert source.index('_ = ' + owner, fit) < out


def test_public_probe_requires_actual_new_factory_and_poison_controls():
    path = ROOT / 'python/mojolearn/tests/test_sequential_native_provider.py'
    source = path.read_text()
    ast.parse(source)
    assert 'model_native_api_v1' in source
    assert 'monkeypatch.setitem' in source
    assert 'assert_native()' in source
    assert 'same_state(' in source
    assert 'pytest.skip' not in source
