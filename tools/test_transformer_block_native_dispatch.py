"""Source contracts, not compiled ABI/runtime qualification."""
from pathlib import Path
from check_generation_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]
NATIVE = (ROOT / 'transformer/native_block_dispatch.mojo').read_text()
PROVIDER = (ROOT / 'providers/python/neural_descriptor_pyapi.mojo').read_text()
PUBLIC = (ROOT / 'bindings/neural_block_public.mojo').read_text()


def test_native_block_closure_has_no_cpython_provider_edge():
    _, problems = audit(ROOT, ('transformer.native_block_dispatch',))
    assert not problems, problems
    assert 'PythonObject' not in NATIVE


def test_exact_fresh_and_backward_numerical_owners_are_preserved():
    for call in ('_transformer_run_session[True](owner[], expanded,',
                 '_transformer_run[True](expanded,',
                 '_transformer_run_session_backward(owner[], a,',
                 '_transformer_backward_run(a,'):
        assert call in NATIVE
    assert NATIVE.count('owner[].clear()') == 2
    assert 'p[6], p[1], 0, p[7], opts)' in NATIVE
    assert 'owner[].busy or owner[].closed' in NATIVE
    assert 'if not owner[].busy or owner[].closed:' in NATIVE


def test_common_descriptor_table_not_new_provider_or_extension_unwrap():
    assert 'List[NativeDescriptorCall]' in NATIVE
    assert 'native_descriptor_report(problem, context, sink)' in NATIVE
    body = PROVIDER.split('def neural_descriptor_reserved_execute(')[1].split('def neural_descriptor_info(')[0]
    assert 'NeuralDescriptorProvider' in body
    assert body.index('native_descriptor_call(owner[].entry, reserve,') < body.index('with GILReleased(Python())')
    assert body.count('native_descriptor_call(owner[].entry, release,') == 2
    assert '_ = session_owner' in body and '_ = provider' in body
    for forbidden in ('getattr(', '__call__', 'PyObject_Call', 'unsafe_get_as_pointer'):
        assert forbidden not in body


def test_public_fresh_session_ephemeral_and_backward_are_hooked():
    assert 'native, session_owner, handle, 0, 1, 2, addresses, params, []' in PUBLIC
    assert 'native, 4, addresses, params, []' in PUBLIC
    assert 'native, session_owner, handle, 0, 1, 3,' in PUBLIC
    assert 'native, 5, _neural_int_words(addresses)' in PUBLIC
    # All old supported entries remain under the off/default branches.
    assert 'e["transformer_session_forward_fresh"]' in PUBLIC
    assert 'e["transformer_forward_fresh"]' in PUBLIC
    assert '_native_function(entries, "transformer_session_backward")' in PUBLIC


def test_connection_carries_named_factory_and_checked_session_admission():
    bridge = (ROOT / 'python/mojolearn/_native_neural_bridge.py').read_text()
    assert '"transformer_block_api_v1", "transformer_native_session_handle_v1"' in bridge
    shell = (ROOT / 'bindings/_mojolearn_transformer.mojo').read_text()
    assert 'm.def_function[transformer_block_api_binding]("transformer_block_api_v1")' in shell
    assert 'NATIVE_TRANSFORMER_BLOCK_API.get_or_create_ptr()' in shell
    causal = (ROOT / 'bindings/causal_lm_public.mojo').read_text()
    assert 'py_setitem(entries, "native_block_operations", block_provider)' in causal
    assert 'code == 1 and vendor != "cpu"' in PUBLIC
