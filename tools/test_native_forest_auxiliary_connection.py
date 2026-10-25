"""Source/connection coverage for forest auxiliary native public doors."""
import ast
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python/mojolearn/_forest_protocol.py'


def methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ForestProtocol')
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
            and node.name in ('apply', 'predict_with_leaves', 'staged_predict', '_forest_auxiliary')}


def test_public_and_private_doors_each_delegate_once_without_python_result_handling():
    for name, method in methods().items():
        statements = method.body[1:]
        assert len(statements) == 1 and isinstance(statements[0], ast.Return)
        assert isinstance(statements[0].value, ast.Call) and ast.unparse(statements[0].value.func) == '_TREES'
        calls = []
        model, x, context, result = (object() for _ in range(4))
        namespace = dict(_TREES_CONTEXT=context, _TREES=lambda *args: calls.append(args) or result)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        actual = namespace[name](model, x)
        assert actual is result and len(calls) == 1
        owner, operation, args, connection = calls[0]
        assert owner is model and connection is context and args[0] is x
        assert operation == 'trees_public_forest_' + ('auxiliary' if name.startswith('_') else name)
        assert args[1:] == ((False, False, False) if name.startswith('_') else ())


def test_all_six_registered_forest_classes_inherit_the_three_public_doors():
    classes = {}
    for file in ('_forest_protocol', 'randomforest', 'extratrees', '_expansion_trees'):
        tree = ast.parse((ROOT / ('python/mojolearn/' + file + '.py')).read_text())
        classes.update({node.name: node for node in tree.body if isinstance(node, ast.ClassDef)})
    test_source = ast.parse((ROOT / 'python/mojolearn/tests/test_native_forest_auxiliary.py').read_text())
    owners = next(ast.literal_eval(node.value) for node in test_source.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'OWNERS' for target in node.targets))

    def defining_class(name, method):
        cls = classes.get(name)
        if cls is None:
            return None
        if any(isinstance(node, ast.FunctionDef) and node.name == method for node in cls.body):
            return name
        return next((found for parent in cls.bases if isinstance(parent, ast.Name)
                     and (found := defining_class(parent.id, method)) is not None), None)

    assert len(owners) == 6
    for name in owners:
        for method in ('apply', 'predict_with_leaves', 'staged_predict'):
            assert defining_class(name, method) == 'ForestProtocol'
    bridge = (ROOT / 'python/mojolearn/_native_trees_bridge.py').read_text()
    loop = next(node for node in ast.walk(ast.parse(bridge)) if isinstance(node, ast.For)
                and isinstance(node.iter, ast.Tuple) and all(isinstance(item, ast.Constant) for item in node.iter.elts)
                and ast.literal_eval(node.iter)[:2] == ('fit', '_fit_with_tree_start'))
    assert {'apply', 'predict_with_leaves', 'staged_predict'} <= set(ast.literal_eval(loop.iter))


def test_existing_import_connector_retains_only_compiled_auxiliary_entry():
    source = ROOT / 'python/mojolearn/_native_trees_bridge.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == '_entries')
    package = ModuleType('_forest_aux_fixture')
    package.__path__ = []
    backend = ModuleType(package.__name__ + '._backend')
    backend._HostBinding = type('HostBinding', (), {})
    package._backend = backend
    binding = ModuleType('native_binding')
    binding.x_trees_forest_auxiliary = len
    binding.python_callback = lambda *args: None
    namespace = dict(__package__=package.__name__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    with patch.dict(sys.modules, {package.__name__: package, backend.__name__: backend}):
        table = namespace['_entries'](binding)
    assert table['x_trees_forest_auxiliary'] is len and 'python_callback' not in table


def test_auxiliary_dispatch_preserves_the_recorded_array_boundary_and_old_validation():
    lifecycle = (ROOT / 'bindings/trees_public_lifecycle_native.mojo').read_text()
    route = lifecycle.split('if operation == "trees_public_forest_apply"', 1)[1].split('var b =', 1)[0]
    assert 'return forest_auxiliary(model, method, args, context)' in route
    assert 'parallel_public_result' not in route and 'sequential_input' not in route
    owner = (ROOT / 'bindings/trees_public_auxiliary.mojo').read_text()
    assert owner.index('this estimator is not fitted yet') < owner.index('classical_admit_value(args, context)')
    assert owner.index('classical_admit_value(args, context)') < owner.index('trees_f32(args[0]')
    assert 'x_trees_forest_auxiliary' in owner and 'NotImplementedError' in owner
    assert 'if leaf_requested else PythonObject(None)' in owner
    assert 'if prefix_requested else PythonObject(None)' in owner
    assert 'if prediction_requested else PythonObject(None)' in owner
    assert 'not forest_kind(model, context).startswith("et_")' in owner
    assert 'native_instance_item(fields, PythonObject(name))' in owner
    assert 'classical_admit_value(value, context)' in owner
    assert '_ = buffers^' in owner and '_ = x' in owner
    assert 'forest_auxiliary_device' not in owner and 'forest_auxiliary_host' not in owner
