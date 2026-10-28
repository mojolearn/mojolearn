"""Structural guards for native Program entry ownership; no runtime execution."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / 'providers/python'


def test_program_methods_do_not_resolve_mutable_instance_callables():
    module = ast.parse((ROOT / 'python/mojolearn/ops.py').read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Program')
    expected = {'__init__', 'run', 'profile', 'run_dynamic', 'inspect', 'run_if',
                'while_loop', 'vjp_if', 'vjp_while', 'fit', 'fit_if', 'save', 'save_control', 'free'}
    checked = set()
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef) or method.name not in expected:
            continue
        body = method.body[1:] if ast.get_docstring(method) else method.body
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        call = body[0].value
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        assert call.func.id == ('_SESSION_CALL' if method.name == '__init__' else '_PROGRAM_CALL')
        assert not any(isinstance(n, ast.Attribute) for n in ast.walk(call))
        checked.add(method.name)
    assert checked == expected


def test_native_request_retains_fields_and_routes_through_session_owner():
    source = (PROVIDERS / 'autograd_program_public_request_pyapi.mojo').read_text()
    assert source.index('program_owner_fields(owner)') < source.index('snapshot["_s"]')
    assert 'Python.tuple(owner, snapshot, fields, operation, args)' in source
    assert 'ops_session_public_call_py(' in source
    assert all(word not in source for word in ('PyObject_Call', 'PyCFunction', '._session_native', '._execute_native'))
    dispatcher = (PROVIDERS / 'autograd_program_public_dispatch_pyapi.mojo').read_text()
    assert 'ops_program_save_fields_py[E, KIND](fields, state,' in dispatcher
    assert 'py_setitem(output_fields, "loss_curve_"' in dispatcher
    assert 'fields["_s"] is not session' in dispatcher


def test_constructor_admits_metadata_and_class_before_allocation():
    source = (PROVIDERS / 'autograd_program_owner_pyapi.mojo').read_text()
    assert source.index('program_native_sequence(info)') < source.index('len(info)')
    constructor = source.split('def program_new_owner(', 1)[1]
    assert constructor.index('Program construction refuses changed Python storage hooks') < constructor.index('b.object.__new__')
    assert 'py_setattr' not in source
    assert 'session._program_class' not in source
    assert 'py_setitem(fields, "_s", session)' in source


def test_both_binding_images_export_public_program_entry():
    for path in ('_mojolearn_x_sequence.mojo', '_mojolearn_x_sequence_host.mojo'):
        source = (ROOT / 'bindings' / path).read_text()
        assert 'm.def_py_c_function(ops_program_public_call_c_binding, "ops_program_public_call")' in source
    surface = (ROOT / 'python/mojolearn/_surface_sequence.py').read_text()
    assert '"ops_program_public_call"' in surface
