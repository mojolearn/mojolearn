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
    source = (ROOT / 'bindings/xseq_neural_native_door.mojo').read_text()
    assert 'py_call_typed' not in source
    assert 'stream.fill_uniform(context' not in source
    # The numeric steps are typed RUN calls (sequence/xseq_neural_run.mojo), not named
    # operations of the descriptor table.
    for name in ('xseq_run_rnn_fit', 'xseq_run_rnn_predict', 'xseq_run_mlp_fit', 'xseq_run_mlp_predict',
                 'xseq_run_argmax', 'xseq_run_curve_min', 'xseq_run_threshold'):
        assert name + '(' in source, name
    assert 'from sequence.xseq_neural_run import' in source
    common = (ROOT / 'bindings/xseq_native_common_door.mojo').read_text()
    assert 'neural_descriptor_reuse(' in common
    assert 'String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)' in common
    assert 'neural_descriptor_execute(provider, MODEL_UNIFORM' in common
    assert 'struct XseqProvider' not in common


def test_native_outputs_preserve_helper_route_switches_and_nan_rule():
    source = (ROOT / 'sequence/model_outputs_device.mojo').read_text()
    # IDN_ALL_OFF and the IDN_HPDEV_*_OFF old paths were deleted 2026-11-15 (docs/TOMBSTONES.md).
    for name in ('HOTPATH_SABOTAGE', 'HPD_MAX_N', 'LRD_MAX_N'):
        assert name in source
    assert 'source[0] == source[0]' in source
    assert 'return model_output_fallback(op, r)' in source
    assert 'model_outputs_host' not in source
    assert 'device_argmax_rows(ex.ctx' in source


def test_public_candidate_probe_has_all_recurrent_cells_and_two_label_shapes():
    source = (ROOT / 'python/mojolearn/tests/test_xseq_native_provider.py').read_text()
    ast.parse(source)
    for name in ('RNNRegressor', 'LSTMRegressor', 'GRURegressor', 'MLPClassifier'):
        assert name in source
    assert 'monkeypatch.setitem' in source and 'assert_native()' in source
    assert 'pytest.skip' not in source


def test_builtin_schedule_reuses_native_spec_and_kernel_directly():
    source = (ROOT / 'bindings/optimizer_schedule_native_door.mojo').read_text()
    operation = source.split('def schedule_run(', 1)[1]
    assert 'entries["lr_schedule_series"]' not in operation
    assert 'entries["lr_schedule_at"]' not in operation
    # One typed evaluation plan (89f667f395) serves the series and the single rate; its fill
    # writes straight into the result storage.
    assert 'optimizer_schedule_evaluation(kind, spec, requested, series)' in operation
    assert operation.count('plan.fill(') >= 2
