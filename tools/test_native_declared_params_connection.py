"""Source/connection contracts for the two declared setter families."""
import ast
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'python/mojolearn/tests/test_native_declared_params_setter.py'
CONNECTOR = ROOT / 'python/mojolearn/_native_declared_params.py'


def families():
    tests = ast.parse(RUNTIME.read_text())
    names = {node.targets[0].id: ast.literal_eval(node.value) for node in tests.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in ('PREP_CLASSES', 'DECOMP_CLASSES')}
    return (('_expansion_prep', '_PrepBase', names['PREP_CLASSES'], 'prep'),
            ('_expansion_decomp', '_Base', names['DECOMP_CLASSES'], 'decomp'))


def setter(module_name, base_name):
    tree = ast.parse((ROOT / ('python/mojolearn/' + module_name + '.py')).read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == base_name)
    return next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == 'set_params')


def test_both_public_setters_are_one_native_call_preserving_keywords():
    for module, base, _, _ in families():
        method = setter(module, base)
        body = method.body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        assert ast.unparse(body[0].value) == '_DECLARED_PARAMS_SET(self, params, _DECLARED_PARAMS_CONTEXT)'
        calls = []
        context, model, result, first, last = (object() for _ in range(5))
        namespace = dict(_DECLARED_PARAMS_CONTEXT=context,
                         _DECLARED_PARAMS_SET=lambda *args: calls.append(args) or result)
        exec(compile(ast.Module(body=[method], type_ignores=[]), module, 'exec'), namespace)
        assert namespace['set_params'](model, first=first, invalid=None, last=last) is result
        assert len(calls) == 1 and calls[0][0] is model and calls[0][2] is context
        assert list(calls[0][1]) == ['first', 'invalid', 'last']
        assert calls[0][1]['first'] is first and calls[0][1]['last'] is last


def test_runtime_matrix_covers_every_inherited_public_setter():
    for module, base, expected, _ in families():
        tree = ast.parse((ROOT / ('python/mojolearn/' + module + '.py')).read_text())
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        public = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets))

        def owner(name):
            cls = classes.get(name)
            if cls is None:
                return None
            if any(isinstance(node, ast.FunctionDef) and node.name == 'set_params' for node in cls.body):
                return name
            return next((found for parent in cls.bases if isinstance(parent, ast.Name)
                         and (found := owner(parent.id)) is not None), None)

        assert {name for name in public if owner(name) == base} == set(expected)
    assert sum(len(row[2]) for row in families()) == 46


def test_actual_connector_retains_schemas_and_owners_without_calling_models():
    package = ModuleType('_declared_setter_fixture')
    package.__path__ = []
    modules = {package.__name__: package}

    def module(name, **values):
        result = ModuleType(package.__name__ + '.' + name)
        vars(result).update(values)
        setattr(package, name, result)
        modules[result.__name__] = result
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError('Python model method or constructor ran')

    metadata = dict(plans={})
    expected = {}
    for name, base_name, classes, style in families():
        owner = module(name)
        base = type(base_name, (), dict(set_params=forbidden))
        setattr(owner, base_name, base)
        owner.__all__ = list(classes)
        for class_name in classes:
            cls = type(class_name, (base,), dict(__init__=forbidden, _parameters=('alpha', 'estimator')))
            setattr(owner, class_name, cls)
            metadata['plans'][cls] = dict(kind='declared', recurse=False, names=cls._parameters)
            expected[cls] = style
        # An explicit override and a public function must not inherit this ABI.
        owner.OwnSetter = type('OwnSetter', (base,), dict(set_params=lambda self: forbidden()))
        owner.utility = forbidden
        owner.__all__ += ['OwnSetter', 'utility']
    package._expansion_prep._ENCODE_CONTEXT = {'prep_encode_scores': {len: 'score'}}
    module('_native_classical_bridge', _numpy_free_context=lambda: {})
    module('_native_owner_lifetime', connect_owner_context=lambda context: context.update(owners_ready=True))
    module('_native_public_attributes', _archive_finalizers=lambda package: [],
           _archive_finalizer_binding=forbidden, _CONTEXT={})
    estimator = type('Estimator', (), dict(__init__=forbidden))
    constructor = dict(mro=(estimator, object))
    registry = {'family_msel_estimators': {estimator: constructor}}
    context = dict(ready=False, plans={}, missing=object())
    function = next(node for node in ast.parse(CONNECTOR.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'connect')
    namespace = dict(__package__=package.__name__, CONTEXT=context)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CONNECTOR), 'exec'), namespace)
    with patch.dict(sys.modules, modules):
        namespace['connect'](package, registry, metadata)
    assert context['ready'] and len(context['plans']) == 46
    assert context['parameter_metadata'] is metadata and context['ownership']['owners_ready']
    assert context['release_plans'][estimator]['constructor'] is constructor
    assert len in context['retained_owners'] and estimator in context['retained_owners']
    for cls, style in expected.items():
        plan = context['plans'][cls]
        assert plan['style'] == style and plan['names'] is cls._parameters
        assert dict(plan['pinned'])['__setattr__'] is object.__setattr__


def test_native_owner_preserves_order_and_uses_existing_storage_and_lifetime_guards():
    source = (ROOT / 'bindings/declared_params_native.mojo').read_text()
    operation = source.split('def _assign(', 1)[1].split('def declared_params_c(', 1)[0]
    loop = operation.split('for i in range(len(pairs)):', 1)[1]
    assert loop.index('native parameter names must be exact strings') < loop.index('names.__contains__')
    assert loop.index('names.__contains__') < loop.index('_release(previous, context)')
    assert loop.index('_release(previous, context)') < loop.index('py_setitem(fields, name, value)')
    assert 'native_instance_item(fields, name)' in loop
    assert '"ValueError: mojolearn: invalid parameter "' in loop
    assert '"ValueError: Invalid "' in loop
    assert 'native_release_weakrefs(item, context["ownership"])' in source
    assert '_constructor_layout(item, fields, plan["constructor"])' in source
    assert 'native_release_admit(item, context["ownership"])' in source
    for forbidden in ('fields.clear', 'fields.pop', '_mojolearn_params_reset', 'model.set_params(', 'model.__init__('):
        assert forbidden not in source
    init = (ROOT / 'python/mojolearn/__init__.py').read_text()
    assert init.index('_connect_params_metadata_classes(_params_metadata_owner') < init.index('_connect_declared_params(_params_metadata_sys')
