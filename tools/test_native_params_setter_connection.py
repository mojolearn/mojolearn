"""Source/connection contracts; compiled public behavior has separate probes."""
import ast
from pathlib import Path
import re
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / 'python/mojolearn/_native_params_setter.py'


def constants():
    tree = ast.parse(CONNECTOR.read_text())
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ('_CLASSES', '_FITTED_FIELDS')}


def test_selected_constructors_are_exact_native_field_assignment_schemas():
    source = ast.parse((ROOT / 'python/mojolearn/_expansion_linear.py').read_text())
    classes = {node.name: node for node in source.body if isinstance(node, ast.ClassDef)}
    helpers = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
    selected = constants()['_CLASSES']
    assert len(selected) == 19

    def fields(function, environment):
        result = {}
        def assign(target, value):
            if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                assert len(target.elts) == len(value.elts)
                for left, right in zip(target.elts, value.elts):
                    assign(left, right)
            else:
                assert isinstance(target, ast.Attribute) and ast.unparse(target.value) == 'self'
                assert isinstance(value, ast.Name), 'new constructor validation/conversion requires a native plan'
                result[target.attr] = environment[value.id]
        for node in function.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assign(target, node.value)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                assert ast.unparse(call.func) == '_glm_init' and not call.keywords
                helper = helpers['_glm_init']
                args = [environment[argument.id] for argument in call.args]
                result.update(fields(helper, dict(zip((a.arg for a in helper.args.args), args))))
            else:
                assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        return result

    for name in selected:
        initializer = next(node for node in classes[name].body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        names = [argument.arg for argument in initializer.args.args + initializer.args.kwonlyargs]
        environment = {name: name for name in names}
        result = fields(initializer, environment)
        assert result == {name: name for name in names if name != 'self'}, name


def test_subset_matches_existing_native_constructor_plans_and_inheritance():
    tree = ast.parse((ROOT / 'python/mojolearn/_native_msel_prep.py').read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == '(name, fit, classifier, weighted)')
    assert tuple(row[0] for row in ast.literal_eval(loop.iter)) == constants()['_CLASSES']
    tree = ast.parse((ROOT / 'python/mojolearn/_expansion_linear.py').read_text())
    for name in ('_LinearClassifierMixin', '_LinearRegressorMixin'):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.unparse(cls.bases[0]) == 'NativeParamsSetter'
    mode = (ROOT / 'python/mojolearn/_mode.py').read_text()
    assert 'replacement = type(self)(**values)' in mode  # remaining families stay visible, unchanged


def test_owned_fitted_state_covers_all_linear_owner_writes():
    owner = (ROOT / 'bindings/x_linear_native.mojo').read_text()
    writes = set(re.findall(r'py_setitem\(fields, "([^"]+)"', owner))
    assert writes <= set(constants()['_FITTED_FIELDS'])
    assert not [line for line in owner.splitlines() if 'py_setitem(fields,' in line
                and not re.search(r'py_setitem\(fields, "', line)]


def test_connection_never_constructs_or_calls_estimators():
    values = constants()
    package = ModuleType('_native_setter_fixture')
    package.__path__ = []
    linear = ModuleType(package.__name__ + '._expansion_linear')
    package._expansion_linear = linear
    bridge = ModuleType(package.__name__ + '._native_classical_bridge')
    bridge._numpy_free_context = lambda: {}
    lifetime = ModuleType(package.__name__ + '._native_owner_lifetime')
    lifetime.connect_owner_context = lambda context: context.update(ownership_connected=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('Python constructor or parameter callback ran')
    registry = {}
    for name in values['_CLASSES']:
        cls = type(name, (), dict(__init__=forbidden, get_params=forbidden, set_params=forbidden))
        setattr(linear, name, cls)
        registry[cls] = dict(init=None, forest=None, initial_fields={}, parameters=('alpha',))
    tree = ast.parse(CONNECTOR.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'connect')
    namespace = dict(values, __package__=package.__name__, _CONTEXT={'plans': {}, 'ready': False})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CONNECTOR), 'exec'), namespace)
    metadata = {'plans': {}}
    with patch.dict(sys.modules, {package.__name__: package, linear.__name__: linear,
                                 bridge.__name__: bridge, lifetime.__name__: lifetime}):
        namespace['connect']({'family_msel_estimators': registry}, metadata)
    assert len(namespace['_CONTEXT']['plans']) == 19
    assert namespace['_CONTEXT']['parameter_metadata'] is metadata
    assert namespace['_CONTEXT']['ready']
    assert namespace['_CONTEXT']['ownership']['ownership_connected']
    for cls, row in namespace['_CONTEXT']['plans'].items():
        assert row['constructor'] is registry[cls]


def test_public_setter_is_one_binding_call_and_registration_has_correct_indent():
    tree = ast.parse(CONNECTOR.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    setter = next(node for node in cls.body if isinstance(node, ast.FunctionDef))
    body = setter.body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert ast.unparse(body[0].value) == '_ENTRY(self, params, _CONTEXT)'
    registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert '\n        module.def_py_c_function(params_setter_c, "params_setter_native_call")\n' in registration
    assert '\n        module.def_py_c_function(params_metadata_c, "params_metadata_native_call")\n' in registration


def test_setter_preserves_references_and_never_invokes_python_estimator_protocols():
    source = (ROOT / 'bindings/params_setter_native.mojo').read_text()
    for forbidden in ('Python.import_module', 'msel_clone_parameter', '.get_params(', '.set_params(', '.__init__(', 'fields.clear'):
        assert forbidden not in source
    assert 'py_setitem(fresh, parameters[i], values[parameters[i]])' in source
    assert 'Py_EnterRecursiveCall' in source and 'Py_LeaveRecursiveCall' in source
    assert '_release_admit(fields[fitted[i]], context, seen)' in source
    assert 'native_release_weakrefs(item, context["ownership"])' in source
    assert 'native_release_admit(item, context["ownership"])' in source


def test_reset_marker_preserves_legacy_user_fields_but_new_fit_wins():
    source = (ROOT / 'bindings/mode_metadata_native.mojo').read_text()
    start = source.index('def _fitted(')
    end = source.index('\ndef _class_fields', start)
    body = source[start:end]
    assert body.index('"_mojolearn_fitted"') < body.index('"_mojolearn_params_reset"') < body.index('name.endswith("_")')
