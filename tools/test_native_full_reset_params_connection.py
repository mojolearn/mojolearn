"""Source/connection checks only; actual setters require the integrated binary."""
import ast
import builtins
import hashlib
import inspect
import json
from pathlib import Path
from types import BuiltinFunctionType, ModuleType
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = ROOT / 'python/mojolearn/_native_full_reset_params.py'
PROOF = json.loads((ROOT / 'tools/native_full_reset_constructor_schemas.json').read_text())
ROWS = PROOF['classes']


def resolved(cls, name, missing):
    return next((vars(base)[name] for base in cls.__mro__ if name in vars(base)), missing)


def fixture():
    """Inert declared signatures, never import product code or native bindings."""
    mode = ModuleType('full_reset_fixture._mode')
    mode.ParamsMixin = type('ParamsMixin', (), {'set_params': lambda self, **params: self})
    attributes = ModuleType('full_reset_fixture._native_public_attributes')
    attributes._resolved = resolved
    attributes._archive_finalizers = lambda package: {}
    attributes._archive_finalizer_binding = lambda *args: None
    attributes._CONTEXT = {}
    host = ModuleType('full_reset_fixture._classical_host')
    package = ModuleType('full_reset_fixture')
    package._classical_host = host
    classes = []
    for row in ROWS:
        definition = ast.parse(row['signature_source'] + ':\n    pass').body[0]
        if row['numeric_mode_injection'] and 'numeric_mode' not in [arg.arg for arg in definition.args.args + definition.args.kwonlyargs]:
            definition.args.kwonlyargs.append(ast.arg(arg='numeric_mode'))
            definition.args.kw_defaults.append(ast.Constant(value=None))
        # Named module constants are inert identity sentinels here. Their
        # actual declarations remain bound by the constructor source proof.
        namespace = {node.id: object() for node in ast.walk(definition)
                     if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                     and not hasattr(builtins, node.id)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), '<inert signature>', 'exec'), namespace)
        classes.append(type(row['declared_class'], (mode.ParamsMixin,), {'__init__': namespace['__init__']}))
    kernel_ridge = next(cls for cls in classes if cls.__name__ == 'KernelRidge')
    host.HostKernelRidge = type('HostKernelRidge', (kernel_ridge,), {'__module__': host.__name__})
    source = ast.parse(CONNECTOR.read_text())
    events = next(ast.literal_eval(n.value) for n in source.body
                  if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '_EVENTS')
    body = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    calls = []
    hooks = next(ast.literal_eval(n.value) for n in source.body
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '_CONSTRUCTOR_HOOKS')
    ns = {'__package__': package.__name__, '_CLASSES': tuple(classes), '_EVENTS': events,
          'BuiltinFunctionType': BuiltinFunctionType, '_ENTRY': lambda *args: calls.append(args),
          '_INIT_ENTRY': lambda *args: calls.append(('init',) + args), '_CONSTRUCTOR_HOOKS': hooks,
          '_CONTEXT': dict(plans={}, missing=object(), ready=False, originals={})}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(CONNECTOR), 'exec'), ns)
    modules = {module.__name__: module for module in (mode, attributes, package, host)}
    return ns, modules, calls


def connected():
    ns, modules, calls = fixture()
    metadata = {'plans': {cls: {'names': tuple(inspect.signature(cls.__init__).parameters)[1:]}
                          for cls in ns['_CLASSES']}}
    with patch.dict('sys.modules', modules):
        ns['install']()
        ns['connect'](metadata, {'plans': {}}, {'plans': {}}, {'family_msel_estimators': {}})
    return ns, modules, calls


def test_install_preserves_77_signatures_defaults_and_one_call_boundary():
    ns, modules, calls = fixture()
    originals = {cls: (cls.__init__, inspect.signature(cls.__init__)) for cls in ns['_CLASSES']}
    with patch.dict('sys.modules', modules):
        ns['install']()
    assert len(ns['_CONTEXT']['plans']) == 77
    for cls, (constructor, signature) in originals.items():
        # The public constructor is now one native call; the original body
        # is retained only as the test oracle, never called by the product.
        assert cls.__init__ is not constructor and ns['_CONTEXT']['originals'][cls] is constructor
        assert not hasattr(cls.__init__, '__wrapped__')
        assert cls.__init__.__qualname__ == cls.__qualname__ + '.__init__'
        current = inspect.signature(cls.__init__)
        assert tuple(current.parameters) == tuple(signature.parameters)
        for name, parameter in signature.parameters.items():
            assert current.parameters[name].kind is parameter.kind
            assert current.parameters[name].default is parameter.default
        plan = ns['_CONTEXT']['plans'][cls]
        assert plan['names'] == tuple(signature.parameters)[1:]
        for name, default in plan['defaults'].items():
            assert default is signature.parameters[name].default
        obj = object.__new__(cls)
        marker = object()
        obj.set_params(z=marker, a=marker)
        assert calls[-1][0] is obj and calls[-1][2] is ns['_CONTEXT']
        assert list(calls[-1][1]) == ['z', 'a'] and calls[-1][1]['z'] is marker
        made = cls(marker, z=marker)
        assert calls[-1][0] == 'init' and calls[-1][1] is made and calls[-1][2] == (marker,)
        assert calls[-1][3] == {'z': marker} and calls[-1][4] is ns['_CONTEXT'] and calls[-1][5] is cls
        names = tuple(signature.parameters)[1:]
        assert plan['positional'] == tuple(n for n in names
            if signature.parameters[n].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD)
        assert set(plan['required_positional'] + plan['required_kwonly']) == {
            n for n in names if signature.parameters[n].default is inspect.Parameter.empty}
        assert plan['rule_fields'] == tuple(dict.fromkeys(e[0] for e in plan['events'] if e[3]))
        assert plan['label'] == constructor.__qualname__
    assert len(calls) == 2 * 77


def test_host_inheritance_stays_on_original_pending_contract():
    ns, modules, _ = fixture()
    host = modules['full_reset_fixture._classical_host'].HostKernelRidge
    original = host.set_params
    with patch.dict('sys.modules', modules):
        ns['install']()
    assert host.set_params is original
    assert host in ns['_CONTEXT']['deferred_host_classes']
    assert host not in ns['_CONTEXT']['plans']
    assert host.__mro__[1].set_params is ns['NativeFullResetParams'].set_params


@pytest.mark.parametrize('kind', ['setter', 'variadic', 'missing_field'])
def test_stale_or_non_field_only_contract_fails_before_installing_any_setter(kind):
    ns, modules, calls = fixture()
    last = ns['_CLASSES'][-1]
    original = ns['_CLASSES'][0].set_params
    if kind == 'setter':
        last.set_params = lambda self, **params: self
    elif kind == 'variadic':
        last.__init__ = lambda self, **params: None
    else:
        ns['_EVENTS'][last.__name__] += (('unknown', 'parameter', 'not_in_signature', False),)
    with patch.dict('sys.modules', modules), pytest.raises(TypeError):
        ns['install']()
    assert ns['_CLASSES'][0].set_params is original and not calls


def test_connection_pins_final_storage_hooks_without_constructing():
    ns, modules, calls = connected()
    assert ns['_CONTEXT']['ready'] and not calls
    assert ns['_CONTEXT']['constructor_hook_types'] == (type(vars(object)['__getattribute__']),
                                                         type(vars(object)['__setattr__']))
    for cls, plan in ns['_CONTEXT']['plans'].items():
        assert plan['mro'] == cls.__mro__
        assert tuple(name for name, _ in plan['constructor_hooks']) == ('__getattribute__', '__getattr__', '__setattr__')
        assert plan['constructor_hooks'][0][1] is object.__getattribute__
        assert plan['constructor_hooks'][1][1] is ns['_CONTEXT']['missing']
        assert all(value is vars(base).get(name) for base, name, value in plan['layout_hooks'])
        assert plan['release_allowed']
        assert plan['field_names'] == tuple(dict.fromkeys(event[0] for event in plan['events']))


def test_connection_rejects_getter_schema_mismatch():
    ns, modules, _ = fixture()
    with patch.dict('sys.modules', modules):
        ns['install']()
        with pytest.raises(TypeError, match='getter schemas differ'):
            ns['connect']({'plans': {cls: {'names': ()} for cls in ns['_CLASSES']}}, {}, {}, {'family_msel_estimators': {}})
    assert not ns['_CONTEXT']['ready']


@pytest.mark.parametrize('recipe,accepted', [(('set_params', (('arg', 0),)), True), (('set_params', (('arg', 1),)), False)])
def test_nested_native_entry_requires_exact_supported_dictionary_abi(recipe, accepted):
    ns, modules, calls = fixture()
    cls = type('NativeChild', (), {})
    plan = {'methods': {'set_params': recipe}, 'entry': len, 'context': object()}
    metadata = {'plans': {c: {'names': tuple(inspect.signature(c.__init__).parameters)[1:]} for c in ns['_CLASSES']}}
    with patch.dict('sys.modules', modules):
        ns['install']()
        if accepted:
            ns['connect'](metadata, {}, {}, {'family_msel_estimators': {cls: plan}})
            assert ns['_CONTEXT']['native_setters'][cls]['entry'] is len
        else:
            with pytest.raises(TypeError, match='dictionary ABI'):
                ns['connect'](metadata, {}, {}, {'family_msel_estimators': {cls: plan}})
    assert not calls


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['declared_class'])
def test_every_schema_is_bound_to_unchanged_constructor_and_helper_source(row):
    for function in row['source_functions']:
        source = ast.parse((ROOT / function['path']).read_text())
        matches = [node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                   and node.name == function['name'] and node.lineno == function['line']]
        assert len(matches) == 1
        digest = hashlib.sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()
        assert digest == function['ast_sha256'], function


def test_schema_events_preserve_repeated_writes_constants_and_mode_wrapper_order():
    ns, _, _ = fixture()
    repeats = constants = 0
    for row in ROWS:
        events = tuple((write['field'], 'parameter' if 'parameter' in write['value'] else 'constant',
                        next(iter(write['value'].values())), write['source'].startswith('NumericModeMixin'))
                       for write in row['writes'])
        assert ns['_EVENTS'][row['declared_class']] == events
        repeats += len(events) != len({event[0] for event in events})
        constants += any(event[1] == 'constant' and not event[3] for event in events)
    assert repeats == 8 and constants == 6
    assert ('stack_method', 'constant', 'auto', False) in ns['_EVENTS']['StackingRegressor']


def test_constructor_resolution_and_mode_wrapper_still_match_the_audited_source():
    from public_runtime_inventory import Index
    index = Index(ROOT)
    for row in ROWS:
        kind, cls = index.resolve_name(row['module'], row['declared_class'])
        assert kind == 'class'
        ancestry = []
        for base in index.mro(cls):
            declarations = []
            for node in base.node.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(target, ast.Name) and target.id == '_mojolearn_native_public' for target in targets):
                        declarations.append(ast.unparse(node))
            ancestry.append(dict(module=base.mod.name, name=base.name,
                bases=[ast.unparse(node) for node in base.node.bases],
                decorators=[ast.unparse(node) for node in base.node.decorator_list],
                native_public_declarations=declarations))
        assert ancestry == row['constructor_ancestry'], row['declared_class']
    wrapper = PROOF['mode_wrapper_source']
    source = ast.parse((ROOT / wrapper['path']).read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == wrapper['class'])
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == wrapper['method'])
    assert hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest() == wrapper['ast_sha256']


def test_full_reset_is_separate_from_existing19_and_all_layout_snapshots_follow_install():
    source = (ROOT / 'python/mojolearn/__init__.py').read_text()
    ordered = ['_install_full_reset_params()', '_connect_mode_metadata_classes(', '_connect_public_attributes(',
               '_connect_native_cv(', '_connect_params_setter(', '_connect_params_metadata_classes(', '_connect_full_reset_params(']
    positions = [source.index(name) for name in ordered]
    assert positions == sorted(positions)
    existing = ast.parse((ROOT / 'python/mojolearn/_native_params_setter.py').read_text())
    names = next(ast.literal_eval(node.value) for node in existing.body if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == '_CLASSES')
    assert len(names) == 19 and not set(names).intersection(row['declared_class'] for row in ROWS)


def test_native_reset_never_invokes_python_constructor_or_clones_parameters():
    source = (ROOT / 'bindings/full_reset_params_native.mojo').read_text()
    for forbidden in ('Python.import_module', '.__init__(', 'clone_parameter', 'msel_native_set_params', 'fields['):
        assert forbidden not in source
    assert source.index('_release_owned(previous, retained, context)') < source.index('py_call(fields.clear)')
    tree = ast.parse(CONNECTOR.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = cls.body[0]
    assert len(method.body) == 2 and ast.unparse(method.body[1]) == 'return _ENTRY(self, params, _CONTEXT)'
    registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert 'from bindings.full_reset_params_native import full_reset_params_c\n' in registration
    assert 'module.def_py_c_function(full_reset_params_c, "full_reset_params_native_call")' in registration


def test_compile_invalidation_is_only_host_tokenizer():
    from development_build import SourceGraph
    groups = json.loads((ROOT / 'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    owners = set()
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            if 'bindings/full_reset_params_native.mojo' in closure['files']:
                owners.add((mode, name))
    assert owners == {('host', 'tokenizer')}


def test_native_constructor_is_one_call_and_never_runs_python_initializers():
    tree = ast.parse(CONNECTOR.read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_initializer')
    wrapper = factory.body[0]
    assert isinstance(wrapper, ast.FunctionDef) and len(wrapper.body) == 1
    assert ast.unparse(wrapper.body[0]) == 'return _INIT_ENTRY(self, args, kwargs, _CONTEXT, owner)'
    source = (ROOT / 'bindings/full_reset_constructor_native.mojo').read_text()
    for forbidden in ('Python.import_module', '.__init__(', '.__setattr__(', 'fields[', 'fields.clear',
                      'native_storage_allocate', 'py_setattr(', 'b.hasattr', 'Python.evaluate'):
        assert forbidden not in source, forbidden
    # Every dropped owner is admitted before the first write.
    assert source.index('_release_owned(previous, retained, context)') < source.index(
        'py_setitem(fields, targets[i], sources[i])')
    # Argument errors precede any admission or state access.
    body = source[source.index('def _construct('):]
    assert body.index('_bind(positional, keywords, plan)') < body.index('_admit(model, owner, plan, context)') < body.index('_fields(model)')
    registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert 'from bindings.full_reset_constructor_native import full_reset_constructor_c\n' in registration
    assert '\n        module.def_py_c_function(full_reset_constructor_c, "full_reset_constructor_native_call")\n' in registration


def test_constructor_compile_invalidation_is_only_host_tokenizer():
    from development_build import SourceGraph
    groups = json.loads((ROOT / 'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    owners = set()
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            if 'bindings/full_reset_constructor_native.mojo' in closure['files']:
                owners.add((mode, name))
    assert owners == {('host', 'tokenizer')}
