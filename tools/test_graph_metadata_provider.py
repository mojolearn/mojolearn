"""Source structure checks; public native behavior probes run only after compilation."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def test_graph_public_constructors_and_helpers_are_one_native_call():
    tree = ast.parse((ROOT / 'python/mojolearn/importers/_graph.py').read_text())
    expected = {'TensorInfo', 'Node', 'GraphIR', 'Resolved', 'UnsupportedModel', '_Native'}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in expected:
            continue
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        assert len(init.body) == 1 and isinstance(init.body[0], ast.Return)
        assert init.body[0].value.func.id == '_READER_NATIVE'
        expected.remove(cls.name)
    assert not expected
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name == '_connect_reader':
            continue
        body = function.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return), function.name
        assert body[0].value.func.id == '_READER_NATIVE'


def test_recursive_record_protocols_are_compiled_descriptors():
    bridge = (ROOT / 'python/mojolearn/importers/_native_reader_bridge.py').read_text()
    assert 'MethodType(entry, (name, context))' in bridge
    assert "entry('record_descriptor', bound, None, context)" in bridge
    source = (ROOT / 'providers/python/autograd_graph_metadata_pyapi.mojo').read_text()
    for symbol in ('PyInstanceMethod_New', 'Py_ReprEnter', 'Py_ReprLeave', 'graph_record_bound', '_admit', '_fields'):
        assert symbol in source
    assert 'Python.import_module' not in source
    assert 'numpy_array_interface' in source and 'value.ctypes' not in source
    dispatch = (ROOT / 'providers/python/autograd_import_frontend_pyapi.mojo').read_text()
    assert 'return graph_record_bound(args).steal_data()' in dispatch


def test_graph_connection_does_not_depend_on_importing_onnx_first():
    graph = (ROOT / 'python/mojolearn/importers/_graph.py').read_text()
    onnx = (ROOT / 'python/mojolearn/importers/onnx.py').read_text()
    assert '_READER_NATIVE, _READER_CONTEXT = _connect_reader()' in graph
    assert 'class _Native:' in graph and 'class _Native:' not in onnx
    assert '_READER_NATIVE, _READER_CONTEXT' in onnx
    assert '_graph._READER_NATIVE' not in onnx
