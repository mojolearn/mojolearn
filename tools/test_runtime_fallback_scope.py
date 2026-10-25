"""Scope controls for the named metadata exception, without product imports."""
import importlib.util
import ast
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch


def fixture():
    package = ModuleType('_fallback_fixture'); package.__path__ = []
    arrays = ModuleType('_fallback_fixture._array')
    class Array:
        def __len__(self):
            return 3
    arrays.Array = Array
    trees = ModuleType('mojolearn._expansion_trees')
    class VotingRegressor:
        def fit(self, x):
            return x
        def set_params(self, x):
            return x
    trees.VotingRegressor = VotingRegressor
    modules = {package.__name__: package, arrays.__name__: arrays, trees.__name__: trees}
    path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_runtime_fallbacks.py'
    spec = importlib.util.spec_from_file_location('_fallback_fixture._runtime_fallbacks', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, modules, Array, VotingRegressor


def test_exception_is_limited_to_exact_connected_owner_and_methods():
    policy, modules, Array, Owner = fixture()
    class Child(Owner):
        pass
    with patch.dict(sys.modules, modules):
        assert policy.approved_metadata_callbacks(Owner().fit)
        assert not policy.approved_metadata_callbacks(Owner().set_params)
        assert not policy.approved_metadata_callbacks(Child().fit)


def test_modified_length_body_is_not_implicitly_authorized():
    policy, modules, Array, Owner = fixture()
    Array.__len__ = lambda self: 4
    with patch.dict(sys.modules, modules):
        assert not policy.approved_metadata_callbacks(Owner().fit)


def test_modified_public_method_is_not_implicitly_authorized():
    policy, modules, Array, Owner = fixture()
    def fit(self, x):
        return x
    Owner.fit = fit
    with patch.dict(sys.modules, modules):
        assert not policy.approved_metadata_callbacks(Owner().fit)


def test_exception_registry_is_imported_only_by_verification():
    package = Path(__file__).resolve().parents[1] / 'python/mojolearn'
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
