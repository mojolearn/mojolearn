"""Native ownership/closure source contracts; not runtime qualification."""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
NATIVE = (ROOT / 'mamba/native_block_dispatch.mojo').read_text()
PUBLIC = (ROOT / 'bindings/neural_block_public.mojo').read_text()
PREFILL = (ROOT / 'bindings/neural_prefill_native.mojo').read_text()


def test_native_closure_has_no_python_or_provider_backedge():
    _, failures = audit(ROOT, ('mamba.native_block_dispatch',))
    assert not failures, failures
    assert 'PythonObject' not in NATIVE


def test_existing_numerical_owners_and_recoverable_failure_are_retained():
    for owner in ('_mamba1_backward_run', '_mamba2_backward_run', '_mamba3_backward_run',
                  '_mamba3_run[True]', '_m3_prefill_run', '_m3_prefill_backward_run'):
        assert owner in NATIVE
    assert 'not owned and String(problem).startswith(_M3_BWD_NONFINITE_DY)' in NATIVE
    assert 'owner[].usable = False' in NATIVE
    assert 'owner[].release()' in NATIVE
    assert 'IDN_MAMBA3_REPORTS_ON_REQUEST' in NATIVE
    assert 'p[3] != owner[].weight_generation' in NATIVE


def test_session_ownership_uses_common_transport_and_exact_admission():
    body = PUBLIC.split('def neural_mamba_session_call(')[1].split('def neural_mamba3_restore')[0]
    assert 'neural_descriptor_handle' in body
    assert 'neural_descriptor_reserved_execute(native, session, handle, 0, 1, operation,' in body
    assert 'unsafe_from_address' not in body
    shell = (ROOT / 'bindings/_mojolearn_mamba.mojo').read_text()
    assert 'session.downcast_value_ptr[Mamba3PrefillSession]()' in shell
    assert 'neural_descriptor_capsule(api[].name, api[].entries)' in shell


def test_all_backward_and_prefill_dispatch_modes_reach_native_table():
    for operation in (7, 9):
        assert f'neural_mamba_session_call(native, fields, entries, {operation},' in PUBLIC
    assert 'neural_descriptor_execute(native, 4,' in PUBLIC
    assert 'neural_descriptor_execute(native, 3 if kind == "mamba2" else 2,' in PUBLIC
    for operation in (6, 8):
        assert f'neural_mamba_session_call(native, fields, entries, {operation},' in PREFILL
    assert 'neural_descriptor_execute(native, 5,' in PREFILL
    assert 'py_call_typed(selected["all_finite_f32"]' not in PUBLIC
