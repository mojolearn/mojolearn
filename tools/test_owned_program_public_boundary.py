"""Source guards for whole-operation native owned-program front doors."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _one_call(method, entry):
    body = method.body[1:] if ast.get_docstring(method) else method.body
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    call = body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == entry
    assert not any(isinstance(item, ast.Attribute) for item in ast.walk(call))


def test_owned_program_methods_do_not_select_instance_entries():
    for name in ('_distributed_program.py', '_partitioned_program.py'):
        tree = ast.parse((ROOT / 'python/mojolearn' / name).read_text())
        cls = next(item for item in tree.body if isinstance(item, ast.ClassDef))
        for method in cls.body:
            if isinstance(method, ast.FunctionDef):
                _one_call(method, '_OWNED_PROGRAM_CALL')


def test_session_factories_and_run_hand_off_without_attributes():
    tree = ast.parse((ROOT / 'python/mojolearn/ops.py').read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'Session')
    wanted = {'run', 'load_partitioned', 'load_distributed'}
    found = set()
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name in wanted:
            _one_call(method, '_SESSION_CALL')
            found.add(method.name)
    assert found == wanted


def test_native_owned_provider_exports_and_cpu_manifest():
    surface = (ROOT / 'python/mojolearn/_surface_sequence.py').read_text()
    for export in ('owned_program_public_api_v1', 'ops_owned_program_connect', 'ops_owned_program_call'):
        assert f'"{export}"' in surface
        for binding in ('_mojolearn_x_sequence.mojo', '_mojolearn_x_sequence_host.mojo'):
            source = (ROOT / 'bindings' / binding).read_text()
            assert f'"{export}"' in source
