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
    deferred = next(ast.literal_eval(n.value) for n in source.body
                    if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '_DEFERRED_MODULES')
    body = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    calls = []
    ns = {'__package__': package.__name__, '_CLASSES': tuple(classes), '_EVENTS': events,
          '_DEFERRED_MODULES': deferred,
          'BuiltinFunctionType': BuiltinFunctionType, '_ENTRY': lambda *args: calls.append(args),
          '_CONTEXT': dict(plans={}, missing=object(), ready=False)}
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
        assert cls.__init__ is constructor
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
    assert len(calls) == 77


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
    for cls, plan in ns['_CONTEXT']['plans'].items():
        assert plan['mro'] == cls.__mro__
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


def deferred_fixture():
    """Eager classes in _CLASSES; the NumPy-backed modules' classes stay unconnected."""
    ns, modules, calls = fixture()
    lazy = {name: module for module, names in ns['_DEFERRED_MODULES'].items() for name in names}
    every = ns['_CLASSES']
    ns['_CLASSES'] = tuple(cls for cls in every if cls.__name__ not in lazy)
    namespaces = {}
    for cls in every:
        if cls.__name__ in lazy:
            module = lazy[cls.__name__]
            namespaces.setdefault(module, {'__name__': 'full_reset_fixture.' + module})[cls.__name__] = cls
    metadata = {'plans': {cls: {'names': tuple(inspect.signature(cls.__init__).parameters)[1:]} for cls in every}}
    order = []
    mode = modules['full_reset_fixture._mode']
    mode._PARAMS_METADATA_CONTEXT = metadata
    modules['full_reset_fixture._native_public_attributes'].connect_classes = (
        lambda package: order.append(('attributes', package.__name__)))
    getters = ModuleType('full_reset_fixture._native_params_metadata')
    getters.connect_classes = lambda context, package: order.append(('getters', context is metadata))
    modules[getters.__name__] = getters
    return ns, modules, calls, namespaces, metadata, order


def test_deferred_modules_are_exactly_the_numpy_backed_rows():
    ns, _, _ = fixture()
    deferred = ns['_DEFERRED_MODULES']
    assert set(deferred) == {'_x_sequence_croston', '_x_sequence_rnn', '_x_sequence_mlp'}
    lazy = [name for names in deferred.values() for name in names]
    assert len(lazy) == len(set(lazy)) == 11
    rows = {row['declared_class']: row['module'] for row in ROWS}
    for module, names in deferred.items():
        for name in names:
            assert rows[name] == 'mojolearn.' + module, (name, rows[name])
    source = ast.parse(CONNECTOR.read_text())
    eager = next(n.value for n in source.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '_CLASSES')
    eager_names = [element.attr for element in eager.elts]
    assert not set(eager_names) & set(lazy)
    assert sorted(eager_names + lazy) == sorted(rows)


def test_connector_imports_no_numpy_module_at_import():
    source = ast.parse(CONNECTOR.read_text())
    for node in source.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            for alias in node.names:
                name = node.module or alias.name
                assert name not in ns_deferred(), name
                path = ROOT / 'python/mojolearn' / (name + '.py')
                if path.exists():
                    module = ast.parse(path.read_text())
                    assert not any(isinstance(n, (ast.Assign, ast.Expr)) and isinstance(n.value, ast.Call)
                                   and ast.unparse(n.value.func) == 'require_numpy' for n in module.body), name


def ns_deferred():
    source = ast.parse(CONNECTOR.read_text())
    return next(ast.literal_eval(n.value) for n in source.body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == '_DEFERRED_MODULES')


@pytest.mark.parametrize('module', sorted(ns_deferred()))
def test_each_deferred_module_connects_itself_as_its_last_statement(module):
    tree = ast.parse((ROOT / 'python/mojolearn' / (module + '.py')).read_text())
    tail = [ast.unparse(node) for node in tree.body[-3:]]
    assert tail == ['from ._native_full_reset_params import connect_module as _connect_full_reset',
                    '_connect_full_reset(globals())', 'del _connect_full_reset']


def test_deferred_module_connects_after_package_in_package_order():
    ns, modules, calls, namespaces, metadata, order = deferred_fixture()
    lazy = [cls for namespace in namespaces.values() for cls in namespace.values() if type(cls) is type]
    originals = {cls: cls.set_params for cls in lazy}
    with patch.dict('sys.modules', modules):
        ns['install']()
        ns['connect'](metadata, {'plans': {}}, {'plans': {}}, {'family_msel_estimators': {}})
        assert ns['_CONTEXT']['ready'] and len(ns['_CONTEXT']['plans']) == 66
        assert all(cls not in ns['_CONTEXT']['plans'] and cls.set_params is originals[cls] for cls in lazy)
        for module, namespace in namespaces.items():
            classes = [cls for cls in namespace.values() if type(cls) is type]
            seen = []
            modules['full_reset_fixture._native_public_attributes'].connect_classes = (
                lambda package, classes=classes, seen=seen: seen.append(
                    ('attributes', all(c.set_params is ns['NativeFullResetParams'].set_params for c in classes))))
            modules['full_reset_fixture._native_params_metadata'].connect_classes = (
                lambda context, package, seen=seen: seen.append(('getters', context is metadata)))
            ns['connect_module'](namespace)
            assert seen == [('attributes', True), ('getters', True)], module
            for cls in classes:
                plan = ns['_CONTEXT']['plans'][cls]
                assert plan['mro'] == cls.__mro__ and plan['release_allowed']
                obj = object.__new__(cls)
                obj.set_params(alpha=1)
                assert calls[-1][0] is obj and calls[-1][2] is ns['_CONTEXT']
    assert len(ns['_CONTEXT']['plans']) == 77
    assert 'pending_classes' not in ns['_CONTEXT']


def test_deferred_module_loaded_during_package_import_finishes_at_connect():
    ns, modules, calls, namespaces, metadata, order = deferred_fixture()
    namespace = namespaces['_x_sequence_croston']
    classes = [cls for cls in namespace.values() if type(cls) is type]
    with patch.dict('sys.modules', modules):
        ns['install']()
        ns['connect_module'](namespace)
        assert not order
        assert all(cls.set_params is ns['NativeFullResetParams'].set_params for cls in classes)
        assert all('mro' not in ns['_CONTEXT']['plans'][cls] for cls in classes)
        ns['connect'](metadata, {'plans': {}}, {'plans': {}}, {'family_msel_estimators': {}})
    assert not order and 'pending_classes' not in ns['_CONTEXT']
    assert all(ns['_CONTEXT']['plans'][cls]['mro'] == cls.__mro__ for cls in classes)
    assert not calls


def test_deferred_module_schema_mismatch_refuses_before_any_setter():
    ns, modules, calls, namespaces, metadata, order = deferred_fixture()
    namespace = namespaces['_x_sequence_rnn']
    classes = [cls for cls in namespace.values() if type(cls) is type]
    originals = {cls: cls.set_params for cls in classes}
    classes[-1].__init__ = lambda self, **params: None
    with patch.dict('sys.modules', modules):
        ns['install']()
        ns['connect'](metadata, {'plans': {}}, {'plans': {}}, {'family_msel_estimators': {}})
        with pytest.raises(TypeError):
            ns['connect_module'](namespace)
    assert all(cls.set_params is originals[cls] and cls not in ns['_CONTEXT']['plans'] for cls in classes[-1:])
    assert all(cls.set_params is originals[cls] for cls in classes) and not calls


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
