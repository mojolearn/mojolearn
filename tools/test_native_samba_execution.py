"""Source contracts for native Samba traversal; runtime is explicitly pending."""
from pathlib import Path
from check_generation_native_boundary import audit
ROOT=Path(__file__).resolve().parents[1]


def test_native_execution_and_extended_primitive_closures_are_python_free():
    _, failures=audit(ROOT,('training.native_samba_execution','training.native_primitive_dispatch'))
    assert not failures, failures


def test_native_layer_orders_error_refresh_compute_and_writeback():
    source=(ROOT/'training/native_samba_execution.mojo').read_text()
    layer=source.split('def forward(mut self, input_address')[1].split('struct NativeSambaModel')[0]
    assert layer.index('self.failure.byte_length()') < layer.index('copy_to(')
    assert layer.index('copy_to(') < layer.index('self.result = self.endpoint.call')
    assert layer.index('self.result = self.endpoint.call') < layer.index('copy_from(')
    assert layer.index('copy_from(') < layer.index('self.completed = True')
    assert 'self.native' not in source  # executor owns plain native fields


def test_public_boundary_uses_native_model_and_preserves_partial_publication():
    source=(ROOT/'providers/python/samba_execution_pyapi.mojo').read_text()
    assert 'AdmittedNativeExecution' in source
    assert 'layer.failure = String(problem)' in source
    assert 'except problem:\n            self.publish()' in source
    assert 'if not layer.completed:' in source
    assert 'if not layer.visited:' in source
    assert 'self.native.forward(' in source
    assert 'layer.refresh_layouts.append(pin.layout.value().copy())' in source
    public=(ROOT/'bindings/samba_inference_native.mojo').read_text()
    assert 'samba_provider_forward(fields, config, entries, ids, state, step' in public
