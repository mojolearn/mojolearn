"""Source/connection checks for the declared-family native constructors; the
public runtime probes need the integrated binary
(python/mojolearn/tests/test_native_field_constructors.py)."""
import ast
import builtins
import inspect
from pathlib import Path
from types import BuiltinFunctionType, ModuleType
from unittest.mock import patch

import pytest

from declared_constructor_schemas import derive

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / 'python/mojolearn/_native_declared_constructors.py'
SCHEMAS, REFUSED = derive(ROOT)


def literal(name):
    tree = ast.parse(CONNECTOR.read_text())
    return next(n.value for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)


def resolved(cls, name, missing):
    return next((vars(base)[name] for base in cls.__mro__ if name in vars(base)), missing)


def fixture():
    """Inert classes carrying each source constructor's signature; no product import."""
    classes = []
    for name in sorted(SCHEMAS):
        row = SCHEMAS[name]
        module, _, owner = row['defining'].rpartition('.')
        source = ast.parse((ROOT / 'python' / (module.replace('.', '/') + '.py')).read_text())
        cls_node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == owner)
        init = next(n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        stub = ast.FunctionDef(name='__init__', args=init.args, body=[ast.Pass()], decorator_list=[],
                               returns=None, type_params=[])
        namespace = {node.id: object() for node in ast.walk(init.args)
                     if isinstance(node, ast.Name) and not hasattr(builtins, node.id)}
        holder = ast.ClassDef(name=owner, bases=[], keywords=[], body=[stub], decorator_list=[], type_params=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[])), '<inert signature>', 'exec'), namespace)
        function = vars(namespace[owner])['__init__']
        classes.append(type(name, (), {'__init__': function}))
    events = ast.literal_eval(literal('_EVENTS'))
    hooks = ast.literal_eval(literal('_CONSTRUCTOR_HOOKS'))
    body = [n for n in ast.parse(CONNECTOR.read_text()).body if isinstance(n, ast.FunctionDef)]
    calls = []
    attributes = ModuleType('declared_fixture._native_public_attributes')
    attributes._resolved = resolved
    attributes._CONTEXT = {}
    ns = {'__package__': 'declared_fixture', '_CLASSES': tuple(classes), '_EVENTS': events,
          '_CONSTRUCTOR_HOOKS': hooks, 'BuiltinFunctionType': BuiltinFunctionType,
          '_ENTRY': lambda *args: calls.append(args),
          '_CONTEXT': dict(plans={}, missing=object(), ready=False, originals={}, release_declared=True)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(CONNECTOR), 'exec'), ns)
    return ns, {attributes.__name__: attributes}, calls


def test_committed_schemas_are_exactly_the_source_derivation():
    events = ast.literal_eval(literal('_EVENTS'))
    assert events == {name: row['events'] for name, row in SCHEMAS.items()}
    listed = [ast.unparse(node).rpartition('.')[2] for node in literal('_CLASSES').elts]
    assert sorted(listed) == sorted(SCHEMAS) and len(listed) == 50
    # Only LabelEncoder (object's own constructor) stays outside.
    assert REFUSED == {'LabelEncoder': 'no source constructor'}
    # The resample results left this connector: each constructor is one call
    # to the resample owner's native result initializer.
    source = ast.parse((ROOT / 'python/mojolearn/resample.py').read_text())
    for name, kind in (('PermutationTestResult', 'permutation'), ('MonteCarloResult', 'monte_carlo')):
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == name)
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        params = ', '.join(a.arg for a in init.args.args[1:])
        assert len(init.body) == 1 and ast.unparse(init.body[0]) == (
            f"return _NATIVE(None, 'family_rs_result_initialize', (self, {kind!r}, {params}), _NATIVE_CONTEXT)")
        assert name not in SCHEMAS
    for row in SCHEMAS.values():
        assert all(event[3] is False for event in row['events'])
        assert all(type(event[2]) in (str, int, float, bool, type(None)) for event in row['events'])


def test_install_replaces_each_constructor_with_one_native_call():
    ns, modules, calls = fixture()
    originals = {cls: (cls.__init__, inspect.signature(cls.__init__)) for cls in ns['_CLASSES']}
    with patch.dict('sys.modules', modules):
        ns['install']()
    for cls, (constructor, signature) in originals.items():
        assert cls.__init__ is not constructor and ns['_CONTEXT']['originals'][cls] is constructor
        current = inspect.signature(cls.__init__)
        assert [(p.name, p.kind) for p in current.parameters.values()] == [(p.name, p.kind) for p in signature.parameters.values()]
        assert all(current.parameters[n].default is p.default for n, p in signature.parameters.items())
        plan = ns['_CONTEXT']['plans'][cls]
        names = tuple(signature.parameters)[1:]
        assert plan['names'] == names
        assert plan['label'] == constructor.__qualname__
        assert plan['rule_fields'] == ()
        for name, default in plan['defaults'].items():
            assert default is signature.parameters[name].default
        marker = object()
        made = cls(marker, k=marker) if plan['positional'] else cls(k=marker)
        assert calls[-1][0] is made and calls[-1][3] is ns['_CONTEXT'] and calls[-1][4] is cls
    assert len(calls) == 50


def test_connect_pins_storage_hooks_and_the_declared_release_context():
    ns, modules, calls = fixture()
    declared = object()
    with patch.dict('sys.modules', modules):
        ns['install']()
        ns['connect'](declared)
    context = ns['_CONTEXT']
    assert context['ready'] and context['declared'] is declared and context['release_declared'] is True
    assert context['constructor_hook_types'] == (type(vars(object)['__getattribute__']), type(vars(object)['__setattr__']))
    for cls, plan in context['plans'].items():
        assert plan['field_names'] == tuple(dict.fromkeys(e[0] for e in plan['events']))
        assert all(value is vars(base).get(name) for base, name, value in plan['field_hooks'])
        assert tuple(name for name, _ in plan['constructor_hooks']) == ('__getattribute__', '__getattr__', '__setattr__')
    assert not calls


def test_stale_schema_fails_before_replacing_any_constructor():
    ns, modules, calls = fixture()
    first = ns['_CLASSES'][0].__init__
    last = ns['_CLASSES'][-1].__name__
    ns['_EVENTS'][last] += (('unknown', 'parameter', 'not_in_signature', False),)
    with patch.dict('sys.modules', modules), pytest.raises(TypeError):
        ns['install']()
    assert ns['_CLASSES'][0].__init__ is first and not calls


def test_constructor_wrapper_is_one_call_and_wired_before_snapshots():
    tree = ast.parse(CONNECTOR.read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_initializer')
    wrapper = factory.body[0]
    assert len(wrapper.body) == 1 and ast.unparse(wrapper.body[0]) == 'return _ENTRY(self, args, kwargs, _CONTEXT, owner)'
    text = (ROOT / 'python/mojolearn/__init__.py').read_text()
    ordered = ['_install_declared_constructors()', '_connect_mode_metadata_classes(', '_connect_public_attributes(',
               '_connect_declared_params(', '_connect_declared_constructors(']
    positions = [text.index(call) for call in ordered]
    assert positions == sorted(positions)
    native = (ROOT / 'bindings/full_reset_constructor_native.mojo').read_text()
    assert 'if py_contains(context, "release_declared"):' in native
    assert '_declared_release(previous[i], context["declared"])' in native
