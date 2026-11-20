"""Scope controls for the frozen fallback list, without product imports."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _policy():
    path = ROOT / 'python/mojolearn/_runtime_fallbacks.py'
    spec = importlib.util.spec_from_file_location('_fallback_fixture_runtime_fallbacks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_list_is_empty():
    policy = _policy()
    assert policy.FALLBACKS == {}
    assert policy.approved_metadata_callbacks(len) == {}


def test_array_length_is_not_a_python_frame():
    # Array.__len__ is a property over functools.partial, not a def.
    tree = ast.parse((ROOT / 'python/mojolearn/_array.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Array')
    defined = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert not defined & {'__len__', '__getitem__', '__eq__', '__ne__', '__repr__', '__bool__',
                          '__iter__', '__reduce__', '__buffer__'}


def test_exception_registry_is_imported_only_by_verification():
    package = ROOT / 'python/mojolearn'
    importers = set()
    for path in package.rglob('*.py'):
        if 'tests' in path.relative_to(package).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or '').endswith('_runtime_fallbacks'):
                importers.add(path.relative_to(package).as_posix())
            elif isinstance(node, ast.Import):
                if any(alias.name.endswith('_runtime_fallbacks') for alias in node.names):
                    importers.add(path.relative_to(package).as_posix())
    assert importers == {'_verify_native_runtime.py'}
