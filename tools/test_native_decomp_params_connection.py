"""Source-only decomposition getter contracts; no product imports or builds."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python/mojolearn/_expansion_decomp.py'
RUNTIME_TEST = ROOT / 'python/mojolearn/tests/test_native_decomp_params.py'


def getter():
    tree = ast.parse(SOURCE.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_Base')
    return next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == 'get_params')


def test_public_getter_delegates_once_to_existing_native_schema():
    body = getter().body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert ast.unparse(body[0].value) == '_PARAMS_METADATA(self, False, _PARAMS_METADATA_CONTEXT)'
    schema = (ROOT / 'python/mojolearn/_native_params_metadata.py').read_text()
    assert "_expansion_decomp._Base.get_params: ('declared', False)" in schema


def test_all_inherited_public_getters_have_runtime_probes():
    tree = ast.parse(SOURCE.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    public = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets))

    def owner(name, member):
        cls = classes.get(name)
        if cls is None:
            return None
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == member:
                return name, node
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == member
                                                   for target in node.targets):
                return name, node
        return next((found for base in cls.bases if isinstance(base, ast.Name)
                     and (found := owner(base.id, member)) is not None), None)

    selected = {name for name in public if (found := owner(name, 'get_params')) and found[0] == '_Base'}
    tests = ast.parse(RUNTIME_TEST.read_text())
    covered = next(ast.literal_eval(node.value) for node in tests.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'CLASSES' for target in node.targets))
    assert selected == set(covered) and len(selected) == 22
    for name in selected:
        parameters = ast.literal_eval(owner(name, '_parameters')[1].value)
        assert isinstance(parameters, tuple) and all(type(value) is str for value in parameters)


def test_ignored_deep_and_model_protocols_are_not_read_by_wrapper():
    calls = []
    context, result = object(), object()

    def entry(*args):
        calls.append(args)
        return result

    class Trap:
        def __bool__(self):
            raise AssertionError('deep must remain ignored')

        def __getattribute__(self, name):
            raise AssertionError('the wrapper must pass the model unchanged')

    model = Trap()
    namespace = dict(_PARAMS_METADATA=entry, _PARAMS_METADATA_CONTEXT=context)
    exec(compile(ast.Module(body=[getter()], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    assert namespace['get_params'](model, Trap()) is result
    assert calls == [(model, False, context)]
