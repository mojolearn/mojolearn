"""Constructor source controls; compilation and public runtime remain separate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_keys_plain import plain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = plain("bindings/causal_lm_construct_door.mojo", (ROOT / "bindings/causal_lm_construct_door.mojo").read_text())


def test_constructor_uses_private_metadata_after_native_calls():
    constructor = SOURCE.split("def causal_model_init_binding", 1)[1].split("def causal_model_read_weights_binding", 1)[0]
    assert constructor.index("var live = _causal_model_fields(mods, model)") < constructor.index("_model_weights_snapshot(")
    assert 'var route = captured["device"]' in constructor
    assert 'var profile = captured["numeric_profile"]' in constructor
    assert 'model.' not in constructor and 'py_setattr(model,' not in constructor
    assert '_causal_namespace(mods, captured)' in constructor
    assert 'causal_lm_admit_binding(mods, admission,' in constructor
    assert constructor.count('_model_construction_admission(mods, live, admitted)') == 2


def test_parallel_prepare_signature_and_numerical_routes_are_preserved():
    signatures = [line for line in SOURCE.splitlines() if line.startswith('def _causal_model_prepare_snapshot(')]
    assert len(signatures) == 2
    assert signatures[0].count(': PythonObject') == 6
    assert signatures[1].count(': PythonObject') == 8
    prepare = SOURCE.split(signatures[1], 1)[1].split('def causal_model_init_binding', 1)[0]
    assert 'return bindings' in prepare
    assert 'model.' not in prepare and 'py_setattr(model,' not in prepare
    # af92251e02: the five numerical routes call the linalg descriptor entry by the route key, not Python entries
    for name in ('TO_BF16', 'FROM_BF16', 'QUANTIZE_INT8', 'DEQUANTIZE_INT8', 'QUANTIZE_INT15'):
        assert f'_model_linalg(mods, key, CAUSAL_LIN_{name},' in SOURCE
    assert '_native_function(mods, linalg,' not in SOURCE
    assert 'causal_lm_check_finite(ptr, Int(py=value.size))' in prepare


def test_live_publication_checks_keys_and_keeps_tied_weight_identity():
    publish = SOURCE.split('def _model_construction_set', 1)[1].split('def _model_construction_admission', 1)[0]
    assert publish.index('neural_string_keys(mods, live)') < publish.index('py_setitem(live, name, published)')
    assert 'generation_array_snapshot(value)' in publish
    assert 'identity = py_call(bi.id, value)' in publish
    assert 'published = py_get(buffers, identity)' in publish
    assert 'published = py_call(bi.list, value)' in publish
    assert 'published = py_call(bi.dict, value)' in publish
    admission = SOURCE.split('def _model_construction_admission', 1)[1].split('def causal_model_prepare', 1)[0]
    assert admission.index('neural_string_keys(mods, live)') < admission.index('py_setitem(live, name, admitted[name])')
    assert '"_causal_entries_owner"' in admission


def test_post_kernel_array_metadata_admission_precedes_shape_read():
    for name in ('_model_shape', '_model_pack', '_model_quantize15'):
        function = SOURCE.split(f'def {name}(', 1)[1].split('\n\ndef ', 1)[0]
        assert function.index('_causal_array_guard()') < function.index('value.shape')
