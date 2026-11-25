"""Native ownership/closure source contracts; not runtime qualification."""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
NATIVE = (ROOT / 'mamba/native_block_dispatch.mojo').read_text()
PUBLIC = (ROOT / 'bindings/neural_block_public_door.mojo').read_text()
PREFILL = (ROOT / 'bindings/neural_prefill_native_door.mojo').read_text()


def test_native_dependencies_resolve():
    _, failures = audit(ROOT, ('mamba.native_block_dispatch',))
    assert not failures, failures
    assert 'PythonObject' not in NATIVE


def test_existing_numerical_owners_and_recoverable_failure_are_retained():
    for owner in ('_mamba1_backward_run', '_mamba2_backward_run', '_mamba3_backward_run',
                  'mamba_block_call_abi(3, expanded', '_m3_prefill_run', '_m3_prefill_backward_run'):
        assert owner in NATIVE
    assert 'if String(problem).startswith(_M3_BWD_NONFINITE_DY):' in NATIVE
    assert 'owner[].usable = False' in NATIVE
    assert 'owner[].release()' in NATIVE
    assert 'IDN_MAMBA3_REPORTS_ON_REQUEST' in NATIVE


def test_session_ownership_is_a_registry_handle():
    # Lane mamba3-prefill-handle (2026-11-12): the prefill session lives in the
    # image's state registry; callers hold the handle owner and its word.
    assert 'PREFILL_SESSIONS.insert(Mamba3PrefillSession())' in NATIVE
    assert 'PREFILL_SESSIONS.acquire(mamba3_prefill_session_of(words[0]))' in NATIVE
    assert 'def neural_mamba_session_call(' not in PUBLIC
    helpers = (ROOT / 'providers/python/mamba3_prefill_session_pyapi_door.mojo').read_text()
    assert 'downcast_value_ptr[Mamba3PrefillHandle]()' in helpers
    shell = (ROOT / 'bindings/_mojolearn_mamba.mojo').read_text()
    assert 'downcast_value_ptr[Mamba3PrefillSession]()' not in shell
    assert '"mamba_prefill_native_handle_v1"' not in shell
    assert 'neural_descriptor_capsule(api[].name, api[].entries)' in shell


def test_all_backward_and_prefill_dispatch_modes_reach_native_table():
    assert 'mamba3_prefill_session_run(owner, NEURAL_MAMBA3_SESSION_BACKWARD,' in PUBLIC
    assert 'mamba3_prefill_session_run(owner, NEURAL_MAMBA3_SESSION_FORWARD,' in PREFILL
    assert 'neural_descriptor_execute(native, 5,' in PREFILL
    assert 'py_call_typed(selected["all_finite_f32"]' not in PUBLIC


def test_owned_snapshot_mechanism_is_gone():
    # NN40 owned weights deleted with their leftovers (lane owned-weights-deletion,
    # 2026-11-25): no owned op, capability bit, restore path or public method remains.
    source = (ROOT/'bindings/mamba_public_native_door.mojo').read_text()
    for name in ('def mamba3_install_owned(', 'def mamba3_export_owned(', 'def mamba3_release_owned(',
                 '_owned_generation', '_owned_restore_pending'):
        assert name not in source, name
    for name in ('_m3_export_weights_run', '_mamba3_run_owned', '_m3_install_weights_run', 'owned_weights',
                 'weight_generation'):
        assert name not in NATIVE, name
    assert 'def neural_mamba3_restore(' not in PUBLIC
    table = (ROOT/'core/neural_entry_table.mojo').read_text()
    for name in ('comptime NEURAL_MAMBA3_OWNED_', 'comptime MAMBA3_CAP_OWNED_WEIGHTS'):
        assert name not in table, name
    host = (ROOT/'mamba/host/native_block_dispatch.mojo').read_text()
    assert 'StateRegistry' not in host
    assert '_prefill_binding' not in source.split('def mamba3_getstate')[1]


def test_resident_decode_families_use_existing_native_owners():
    from check_generation_native_boundary import audit
    _, failures = audit(ROOT, ('mamba.native_decode_dispatch',))
    assert not failures, failures
    native=(ROOT/'mamba/native_decode_dispatch.mojo').read_text()
    for family in (1,2,3):
        assert f'Mamba{family}DecodeSession' in native
        for stage in ('open','step','export'):
            assert f'_m{family}_session_{stage}_run(owner[]' in native
    assert '_m1_session_load_run(owner[]' in native
    assert '_m2_session_load_run(owner[]' in native
    assert '_m3_session_copy_in(owner[]' in native
    assert 'owner[].usable = False' in native
    assert 'owner[].release()' in native
    boundary=(ROOT/'bindings/mamba_public_native_door.mojo').read_text()
    assert 'var provider: PythonObject' in boundary
    assert 'var device_address: Int' in boundary
    for operation in (3,4,5,6):
        assert f'base + {operation},' in boundary
    assert 'neural_descriptor_execute(h[].provider, 27 + 8 * (h[].plan.kind - 1)' in boundary
