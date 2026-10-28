"""Source guards for Program serialization admission; no runtime execution."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'providers/python/autograd_program_save_pyapi.mojo'


def section(name):
    return SOURCE.read_text().split('def '+name+'(', 1)[1].split('\n\ndef ', 1)[0]


def test_save_reads_only_admitted_owner_storage():
    source = SOURCE.read_text()
    for forbidden in ('program._s', 'program._id', 'b.hasattr', 'b.getattr', 'Python.import_module'):
        assert forbidden not in source
    assert '_fields(program, layout)' in section('_save_fields')
    assert 'py_type(program) is layout["cls"]' in section('_save_fields')
    assert 'ops_session_fields(fields["_s"])' in source
    assert source.index('neural_descriptor_resource_handle(') < source.index('var s = ops_session[')
    assert source.index('var input_names = _names(') < source.index('program_save_verified[E, KIND](')


def test_names_are_passive_and_keep_unnamed_count_fallback():
    source = section('_names')
    assert source.index('py_type(entries) is not b.list') < source.index('len(entries)')
    assert source.index('py_type(row) is not b.list') < source.index('len(row)')
    assert source.index('py_type(row[0]) is not b.str') < source.index('String(py=row[0])')
    assert 'if not py_contains(fields, attribute):\n        return out^' in source
    assert 'if len(entries) != count:\n        return out^' in source


def test_save_provenance_is_import_connected_and_exact():
    source = section('_save_provenance')
    assert 'py_type(version) is not b.str or py_type(platform) is not b.str' in source
    bridge = (ROOT/'python/mojolearn/_native_program_save_bridge.py').read_text()
    assert 'system.sysname + \' \' + system.machine' in bridge
    assert 'layouts=(_layout(cls),), version=__version__' in bridge
    ops = ast.parse((ROOT/'python/mojolearn/ops.py').read_text())
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name) and node.value.func.id == '_connect_program_save'
               for node in ops.body)
    importer = (ROOT/'python/mojolearn/importers/_native_import_bridge.py').read_text()
    assert 'connect_imported(imported)' in importer
