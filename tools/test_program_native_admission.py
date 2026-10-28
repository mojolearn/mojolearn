"""Source checks for provider admission; no compiled runtime is executed."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_program_reuses_shared_registry_before_metadata_or_conversion():
    arrays = (ROOT / 'providers/python/autograd_program_arrays_pyapi.mojo').read_text()
    assert 'classical_connect_numpy(admission)' in arrays
    assert 'sequential_admit_value(value, state[].admission)' in arrays
    assert 'native_array_ndarray_view(value, state[].admission)' in arrays
    source = (ROOT / 'providers/python/autograd_program_pyapi.mojo').read_text()
    entry = source.split('def program_input(', 1)[1].split('\ndef ', 1)[0]
    assert entry.index('program_native_input(value)') < entry.index('array_input(a,')
    assert 'hasattr' not in entry
    scalar = source.split('def _sequence_values(', 1)[1].split('\ndef ', 1)[0]
    assert scalar.index('program_numeric_scalar(a)') < scalar.index('Float64(py=a)')
    assert 'hasattr' not in scalar


def test_embedding_reuses_admitted_descriptor_instead_of_dtype_hook():
    source = (ROOT / 'providers/python/autograd_ops_frontend_pyapi.mojo').read_text()
    assert 'hasattr, args[1], "dtype"' not in source
    assert 'input.src.itemsize == 4 else 10' in source
    assert '_ = args  # retain original external array exporters' in source


def test_common_context_is_connected_at_import_not_per_operation():
    source = (ROOT / 'python/mojolearn/ops.py').read_text()
    module = ast.parse(source)
    connect = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_connect_session_mode')
    text = ast.get_source_segment(source, connect)
    assert 'shared_context()' in text
    assert 'admission = shared_context()' in text
    assert 'result.initialize(None, None, None, result.native_module, None, admission)' in text
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name in {'Session', 'Program'}:
            assert 'shared_context' not in ast.get_source_segment(source, node)


def test_candidate_probe_contains_protocol_negative_controls_and_native_array():
    source = (ROOT / 'python/mojolearn/tests/test_program_native_admission.py').read_text()
    ast.parse(source)
    for token in ('__float__', '__getattribute__', 'Array.from_list', 'program.run', 'assert_native()'):
        assert token in source
    assert 'pytest.skip' not in source
