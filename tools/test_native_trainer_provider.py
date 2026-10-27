"""Source invariants; runtime qualification is the separately retained public probe."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_small_mlp_uses_shared_provider_and_native_owners():
    src=(ROOT/'bindings/small_mlp_native.mojo').read_text()
    assert 'neural_descriptor_reuse' in src and 'neural_descriptor_execute' in src
    assert 'struct SmallMlpProvider' not in src
    operation=src.split('def _mode_entries(')[1].split('def _mode_name(')[0]
    assert 'entries["small_mlp_api_v1"]' in operation
    assert 'MlpOperationMetadata(entries, provider, String(py=mode))' in operation
    batch=src.split('def _batch_in(')[1].split('def _owned_rows(')[0]
    assert batch.count('= _view(')==1 and batch.count('= _shape(')==1
    assert '_words_from_view(value, view, shape,' in batch
    table=(ROOT/'training/small_mlp_dispatch.mojo').read_text()
    for forbidden in ('PythonObject', 'std.python', 'bindings.', 'providers.'):
        assert forbidden not in table
    for owner in ('mlp_train_step_host(', 'neural_gemm_native(', 'mlp_bias_activation_host(',
                  'mlp_relu_backward_host(', 'mlp_sum_rows_host(', 'identical_ce_loss_host(', 'per_call_step('):
        assert owner in table


def test_neural_gemm_both_doors_share_native_transport():
    door=(ROOT/'bindings/neural_gemm_boundary.mojo').read_text()
    assert 'neural_gemm_native(aa, ba, ca, shape)' in door
    assert 'identical_gemm_into(' not in door
    owner=(ROOT/'training/neural_gemm_native.mojo').read_text()
    assert owner.count('identical_gemm_into(')==1
    assert 'gemm_up_f32(' in owner and 'gemm_down_f32(' in owner
    assert 'ctx.synchronize()' in owner and 'lease.release()' in owner


def test_small_mlp_rejects_hooks_before_admission():
    src=(ROOT/'bindings/small_mlp_native.mojo').read_text().split('def small_mlp_call(')[1]
    assert src.index('sequential_admit_value(args,') < src.index('return small_mlp_create(args)')
