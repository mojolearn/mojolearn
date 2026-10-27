"""Source-only checks for numeric recurrent/MLP execution admission."""
import ast
from pathlib import Path

from check_neural_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]


def test_xseq_whole_model_and_consumed_output_closures_are_native():
    result = audit(roots=('sequence.neural_model_dispatch', 'sequence.model_outputs_device',
                          'sequence.model_outputs_host', 'core.xseq_marshal'))
    assert result['passed'], result['failures']


def test_xseq_numeric_calls_share_existing_provider_and_engine_table():
    source = (ROOT / 'bindings/xseq_neural_native.mojo').read_text()
    assert 'py_call_typed' not in source
    assert 'stream.fill_uniform(context' not in source
    for name in ('RNN_FIT', 'RNN_PREDICT', 'MLP_FIT', 'MLP_PREDICT', 'MODEL_LABELS',
                 'MODEL_ARGMAX', 'MODEL_MIN', 'MODEL_THRESHOLD'):
        assert name in source
    common = (ROOT / 'bindings/xseq_native_common.mojo').read_text()
    assert 'neural_descriptor_reuse(' in common
    assert 'String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)' in common
    assert 'neural_descriptor_execute(provider, MODEL_UNIFORM' in common
    assert 'struct XseqProvider' not in common


def test_native_outputs_preserve_helper_route_switches_and_nan_rule():
    source = (ROOT / 'sequence/model_outputs_device.mojo').read_text()
    for name in ('MOJOLEARN_IDN_ALL_OFF', 'MOJOLEARN_IDN_HPDEV_REDUCE_OFF',
                 'MOJOLEARN_IDN_HPDEV_ELEM_OFF', 'MOJOLEARN_IDN_HPDEV_INIT_OFF',
                 'HOTPATH_SABOTAGE', 'HPD_MAX_N', 'LRD_MAX_N'):
        assert name in source
    assert 'source[0] == source[0]' in source
    assert 'return model_output_host(op, r)' in source
    assert 'device_argmax_rows(ex.ctx' in source


def test_public_candidate_probe_has_all_recurrent_cells_and_two_label_shapes():
    source = (ROOT / 'python/mojolearn/tests/test_xseq_native_provider.py').read_text()
    ast.parse(source)
    for name in ('RNNRegressor', 'LSTMRegressor', 'GRURegressor', 'MLPClassifier'):
        assert name in source
    assert 'monkeypatch.setitem' in source and 'assert_native()' in source
    assert 'pytest.skip' not in source
