"""Constructor source controls; compilation and public runtime remain separate."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "bindings/causal_lm_construct.mojo").read_text()


def test_constructor_uses_private_metadata_after_native_calls():
    constructor = SOURCE.split("def causal_model_init_binding", 1)[1].split("def causal_model_read_weights_binding", 1)[0]
    assert constructor.index("var live = _causal_model_fields(model)") < constructor.index("_model_weights_snapshot(")
    assert 'var route = captured["device"]' in constructor
    assert 'var profile = captured["numeric_profile"]' in constructor
    assert 'model.' not in constructor and 'py_setattr(model,' not in constructor
    assert '_causal_namespace(captured)' in constructor
    assert 'causal_lm_admit_binding(admission,' in constructor
    assert constructor.count('_model_construction_admission(live, admitted)') == 2


def test_parallel_prepare_signature_and_numerical_routes_are_preserved():
    signatures = [line for line in SOURCE.splitlines() if line.startswith('def _causal_model_prepare_snapshot(')]
    assert len(signatures) == 2
    assert signatures[0].count(': PythonObject') == 6
    assert signatures[1].count(': PythonObject') == 8
    prepare = SOURCE.split(signatures[1], 1)[1].split('def causal_model_init_binding', 1)[0]
    assert 'return bindings' in prepare
    assert 'model.' not in prepare and 'py_setattr(model,' not in prepare
    for name in ('to_bf16', 'from_bf16', 'quantize_int8', 'dequantize_int8', 'quantize_int15'):
        assert f'_native_function(linalg, "{name}")' in SOURCE
    assert 'causal_lm_check_finite(ptr, Int(py=value.size))' in prepare


def test_live_publication_checks_keys_and_keeps_tied_weight_identity():
    publish = SOURCE.split('def _model_construction_set', 1)[1].split('def _model_construction_admission', 1)[0]
    assert publish.index('neural_string_keys(live)') < publish.index('py_setitem(live, name, published)')
    assert 'generation_array_snapshot(value)' in publish
    assert 'identity = py_call(bi.id, value)' in publish
    assert 'published = py_get(buffers, identity)' in publish
    assert 'published = py_call(bi.list, value)' in publish
    assert 'published = py_call(bi.dict, value)' in publish
    admission = SOURCE.split('def _model_construction_admission', 1)[1].split('def causal_model_prepare', 1)[0]
    assert admission.index('neural_string_keys(live)') < admission.index('py_setitem(live, name, admitted[name])')
    assert '"_causal_entries_owner"' in admission


def test_post_kernel_array_metadata_and_loop_annotations_are_checked():
    for name in ('_model_shape', '_model_pack', '_model_quantize15'):
        function = SOURCE.split(f'def {name}(', 1)[1].split('\n\ndef ', 1)[0]
        assert function.index('_causal_array_guard()') < function.index('value.shape')
    spec = importlib.util.spec_from_file_location('constructor_loop_guard', ROOT / 'tools/hooks/no_compute_in_shells.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker.scan(SOURCE) == []
