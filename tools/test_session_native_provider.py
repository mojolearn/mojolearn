"""Source-only Session registry/provider contracts; no runtime qualification."""
import ast
from pathlib import Path

from check_neural_native_boundary import audit

ROOT = Path(__file__).resolve().parents[1]


def test_session_constructor_no_longer_calls_packed_open_or_mode():
    source = (ROOT / 'providers/python/autograd_session_frontend_pyapi.mojo').read_text()
    assert 'entry.open' not in source and 'entry.numeric_mode' not in source
    assert 'neural_descriptor_execute(provider, SESSION_OPEN' in source
    assert 'neural_descriptor_adopt(provider, sid, SESSION_CLOSE, scope^)' in source
    assert 'neural_descriptor_entry(module, String(symbols[i]))' in source
    assert 'py_setattr' not in source


def test_native_session_registry_cannot_import_provider_or_python():
    result = audit(roots=('autograd.ops_api', 'sequence.model_device_dispatch', 'sequence.model_host_dispatch'))
    assert result['passed'], result['failures']
    for target, kind in [('device', 'OPS_DEVICE'), ('host', 'OPS_HOST')]:
        source = (ROOT / f'sequence/model_{target}_dispatch.mojo').read_text()
        assert source.index('if op == SESSION_OPEN:') < source.index('var ex = ')
        assert f'options[1] != {kind}' in source
        assert 'options[0] != ops_session_registry_token()' in source
        assert 'options[1] != Int(GLOBAL_NUMERIC_MODE)' in source


def test_generic_resource_pins_provider_and_closes_with_admitted_context():
    source = (ROOT / 'providers/python/neural_descriptor_pyapi.mojo').read_text()
    resource = source.split('struct NativeDescriptorResource(', 1)[1]
    assert 'var provider: PythonObject' in resource
    assert 'self.handle = 0' in resource
    assert 'def __deinit__' in resource
    assert 'self.context, List[Float64](), False' in resource
    assert 'owner[].context[i] != context[i]' in resource
    assert 'py_call(' not in resource
    assert 'ops_session' not in resource  # common ownership, no engine-specific provider


def test_public_handle_is_checked_against_native_owner_before_execution():
    source = (ROOT / 'providers/python/autograd_ops_frontend_pyapi.mojo').read_text()
    method = source.split('def ops_session_call_py[', 1)[1]
    assert method.index('neural_descriptor_resource_handle(') < method.index('var s = ops_session[')
    assert 'admitted_id != Int(py=fields["_id"])' in method
    assert 'neural_descriptor_resource_close(fields["_lifetime"], resource_scope)' in method


def test_public_probe_has_cleanup_poisoned_entry_and_owner_checks():
    source = (ROOT / 'python/mojolearn/tests/test_session_native_provider.py').read_text()
    ast.parse(source)
    for token in ('assert_native()', 'numeric_mode', 'entry.open', 'weakref.ref', 'ops_shape', '_lifetime', '__setattr__'):
        assert token in source
    assert 'pytest.skip' not in source


def test_imported_constructor_shares_public_session_connection_and_owner():
    bridge = (ROOT / 'python/mojolearn/importers/_native_import_bridge.py').read_text()
    native = (ROOT / 'providers/python/autograd_import_frontend_pyapi.mojo').read_text()
    assert 'connections=ops._SESSION_CONNECTIONS' in bridge
    assert 'session=session, program=program' in bridge
    assert 'ops_session_initialize_py(session, PythonObject(False), mode, context["connections"], context["backend"], context["program"])' in native
