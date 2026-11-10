"""Source-only native lifecycle contracts; no execution/bitwise qualification."""
from pathlib import Path
from check_generation_native_boundary import audit
ROOT = Path(__file__).resolve().parents[1]


def test_engine_endpoint_dependencies_resolve():
    _, failures = audit(ROOT, ('core.native_execution',))
    assert not failures, failures


def test_common_invocation_reads_only_admitted_native_state():
    source = (ROOT/'providers/python/native_execution_pyapi.mojo').read_text()
    invocation = source.split('def _native_execution_finish')[1].split('struct AdmittedNativeExecution')[0]
    for forbidden in ('py_call', 'PythonObject', 'downcast_value_ptr', 'getattr', 'import_module'):
        assert forbidden not in invocation
    assert 'with GILReleased(Python()):' in invocation
    assert invocation.index('lease.counted(lease.reserve') < invocation.index('result = native_descriptor_call')
    assert 'operation_lock_acquire(lease.lock)' in invocation
    assert 'finally:' in invocation and 'operation_lock_release(lease.lock)' in invocation
    assert 'self.lease = List[NativeExecutionLease]()' in source


def test_transformer_admin_and_carried_calls_reuse_original_numerical_owner():
    source = (ROOT/'transformer/native_block_dispatch.mojo').read_text()
    assert 'return transformer_session_forward_native(owner[], expanded, p, opts)' in source
    assert 'owner[].closed = True' in source
    # Op 12 (the session tape's invalidation) went with the tape API
    # (TRANSFORMER_FORWARD_TAPE_API, docs/TOMBSTONES.md); 13 is the one admin op.
    assert 'invalidate_tape' not in source.split('def transformer_block_v1')[0].replace('invalidate-tape', '')
    boundary = (ROOT/'bindings/neural_block_public.mojo').read_text()
    assert 'causal_transformer_admin(mods, fields["_session_binding"], fields["_native_session"], 13)' in boundary
    assert 'causal_transformer_admin(mods, entries, fields["_native_session"], 12)' not in boundary
