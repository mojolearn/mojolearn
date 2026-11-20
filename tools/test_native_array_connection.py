"""Source/connection checks for the native Array operations; the runtime
probes need the integrated binary (python/mojolearn/tests/test_native_array_operations.py)."""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_keys_plain import plain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARRAY = ROOT / 'python/mojolearn/_array.py'
# Array's Python definitions, the oracles, live with the tests (lane
# p5-cpu-route-relabel-and-dead-code, 2026-11-09).
REFERENCE = ROOT / 'python/mojolearn/tests/_array_reference.py'
NATIVE = ROOT / 'bindings/array_native_door.mojo'

OPS = {'__init__': 'init', 'flags': 'flags', '__array_interface__': 'interface', '__array__': 'array',
       '__buffer__': 'buffer', '__reduce__': 'reduce', 'tobytes': 'tobytes', 'tolist': 'tolist',
       'copy': 'copy', 'astype': 'astype', 'reshape': 'reshape', 'ravel': 'ravel', '__len__': 'len',
       '__iter__': 'iter', '__repr__': 'repr', '__bool__': 'bool', '__getitem__': 'getitem',
       '__eq__': 'eq', '__ne__': 'ne', 'min': 'min', 'max': 'max', 'sum': 'sum', 'argmax': 'argmax',
       'from_list': 'from_list'}


def classes(path=ARRAY):
    tree = ast.parse(path.read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _bound_assignments(path=ARRAY):
    """`Array.<name> = _bound("op")` and `Array.<name> = property(functools.partial(_ARRAY_NATIVE, "op",
    _ARRAY_CONTEXT), ...)` below the class: the protocol slots and properties that are C partials over the native
    entry (no Python frame). Returns {name: (kind, op, context expression)}."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and ast.unparse(node.targets[0].value) == 'Array' and isinstance(node.value, ast.Call)):
            continue
        call, name = node.value, node.targets[0].attr
        if ast.unparse(call.func) == '_bound':
            found[name] = ('bound', call.args[0].value, '_ARRAY_CONTEXT')
        elif ast.unparse(call.func) == 'property' and isinstance(call.args[0], ast.Call):
            partial = call.args[0]
            if ast.unparse(partial.func) == 'functools.partial' and ast.unparse(partial.args[0]) == '_ARRAY_NATIVE':
                found[name] = ('property', partial.args[1].value, ast.unparse(partial.args[2]))
    return found


def test_every_public_operation_is_one_native_call():
    """Each public Array operation is either a method whose whole body is `return _ARRAY_NATIVE(op, self,
    _ARRAY_CONTEXT, ...)`, or a C partial over the native entry assigned below the class (`_bound(op)`, or a
    `property` over `functools.partial(_ARRAY_NATIVE, op, _ARRAY_CONTEXT)`); the native door handles every op."""
    array = classes()['Array']
    methods = {n.name: n for n in array.body if isinstance(n, ast.FunctionDef)}
    assigned = _bound_assignments()
    bound = ast.parse(ARRAY.read_text())
    helper = next(n for n in bound.body if isinstance(n, ast.FunctionDef) and n.name == '_bound')
    assert 'functools.partial(functools.partial, _ARRAY_NATIVE, op, _ARRAY_CONTEXT)' in ast.unparse(helper)
    # the door dispatches on integer operation codes; its by-name form names each arm
    native = plain('bindings/array_native_door.mojo', NATIVE.read_text())
    for name, op in OPS.items():
        if name in assigned:
            assert name not in methods, name  # one definition only
            _, got, context = assigned[name]
            assert got == op and context == '_ARRAY_CONTEXT', name
        else:
            body = [s for s in methods[name].body
                    if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            assert len(body) == 1 and isinstance(body[0], ast.Return), name
            call = body[0].value
            assert ast.unparse(call.func) == '_ARRAY_NATIVE' and call.args[0].value == op, name
            assert ast.unparse(call.args[2]) == '_ARRAY_CONTEXT', name
        assert f'op == "{op}"' in native, op


def _names_used(path):
    """Identifiers a module actually references (imports, names, attributes); comments and strings excluded."""
    used = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            used.update((node.module or '').split('.'))
            used.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                used.update(a.name.split('.'))
    return used


def test_reference_definitions_are_kept_only_as_oracles():
    reference = classes(REFERENCE)['_ArrayReference']
    names = {n.name for n in reference.body if isinstance(n, ast.FunctionDef)}
    assert names == set(OPS)
    assert '_ArrayReference' not in classes()
    product = ROOT / 'python/mojolearn'
    users = set()
    for path in product.rglob('*.py'):
        if 'tests' in path.relative_to(product).parts:
            continue
        if _names_used(path) & {'_ArrayReference', '_array_reference'}:  # a comment pointing at it is no use
            users.add(path.relative_to(product).as_posix())
    assert users == set()
    text = REFERENCE.read_text()
    assert text.count('_ArrayReference.') == 4  # its docstring and the three internal oracle rewires


def test_registration_and_compile_invalidation():
    registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert 'from bindings.array_native_door import array_native_c\n' in registration
    assert '\n        module.def_py_c_function(array_native_c, "array_native_call")\n' in registration
    from development_build import SourceGraph
    groups = json.loads((ROOT / 'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    owners = set()
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            if 'bindings/array_native_door.mojo' in closure['files']:
                owners.add((mode, name))
    # 2026-11-07 host split (docs/build/HOT_GROUPS.md section 6): the small
    # MLP's item reader (bindings/small_mlp_items_door.mojo) imports it too, and the
    # small MLP entry is now the multi-device host binding's.
    assert owners == {('host', 'tokenizer'), ('host', 'runtime_parallel')}


def test_native_owner_runs_no_package_python():
    code = '\n'.join(line.split('#', 1)[0] for line in NATIVE.read_text().splitlines())
    for forbidden in ('Python.import_module', 'Python.evaluate', '_ArrayReference', '_buffer._native',
                      '_flatten(', '_RowIter'):
        assert forbidden not in code, forbidden
