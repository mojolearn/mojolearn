"""Source/connection checks for the native Array operations; the runtime
probes need the integrated binary (python/mojolearn/tests/test_native_array_operations.py)."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARRAY = ROOT / 'python/mojolearn/_array.py'
NATIVE = ROOT / 'bindings/array_native.mojo'

OPS = {'__init__': 'init', 'flags': 'flags', '__array_interface__': 'interface', '__array__': 'array',
       '__buffer__': 'buffer', '__reduce__': 'reduce', 'tobytes': 'tobytes', 'tolist': 'tolist',
       'copy': 'copy', 'astype': 'astype', 'reshape': 'reshape', 'ravel': 'ravel', '__len__': 'len',
       '__iter__': 'iter', '__repr__': 'repr', '__bool__': 'bool', '__getitem__': 'getitem',
       '__eq__': 'eq', '__ne__': 'ne', 'min': 'min', 'max': 'max', 'sum': 'sum', 'argmax': 'argmax',
       'from_list': 'from_list'}


def classes():
    tree = ast.parse(ARRAY.read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def test_every_public_operation_is_one_native_call():
    array = classes()['Array']
    methods = {n.name: n for n in array.body if isinstance(n, ast.FunctionDef)}
    native = NATIVE.read_text()
    for name, op in OPS.items():
        body = [s for s in methods[name].body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        assert len(body) == 1 and isinstance(body[0], ast.Return), name
        call = body[0].value
        assert ast.unparse(call.func) == '_ARRAY_NATIVE' and call.args[0].value == op, name
        assert ast.unparse(call.args[2]) == '_ARRAY_CONTEXT', name
        assert f'op == "{op}"' in native, op


def test_reference_definitions_are_kept_only_as_oracles():
    reference = classes()['_ArrayReference']
    names = {n.name for n in reference.body if isinstance(n, ast.FunctionDef)}
    assert names == set(OPS)
    product = ROOT / 'python/mojolearn'
    users = set()
    for path in product.rglob('*.py'):
        if 'tests' in path.relative_to(product).parts:
            continue
        if '_ArrayReference' in path.read_text():
            users.add(path.relative_to(product).as_posix())
    assert users == {'_array.py'}
    text = ARRAY.read_text()
    assert text.count('_ArrayReference.') == 4  # its docstring and the three internal oracle rewires


def test_registration_and_compile_invalidation():
    registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert 'from bindings.array_native import array_native_c\n' in registration
    assert '\n        module.def_py_c_function(array_native_c, "array_native_call")\n' in registration
    from development_build import SourceGraph
    groups = json.loads((ROOT / 'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    owners = set()
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            if 'bindings/array_native.mojo' in closure['files']:
                owners.add((mode, name))
    assert owners == {('host', 'tokenizer')}


def test_native_owner_runs_no_package_python():
    code = '\n'.join(line.split('#', 1)[0] for line in NATIVE.read_text().splitlines())
    for forbidden in ('Python.import_module', 'Python.evaluate', '_ArrayReference', '_buffer._native',
                      '_flatten(', '_RowIter'):
        assert forbidden not in code, forbidden
