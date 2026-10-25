"""Source-only prep getter contracts; native execution has separate probes."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python/mojolearn/_expansion_prep.py'
RUNTIME_TEST = ROOT / 'python/mojolearn/tests/test_native_prep_params.py'


def getter():
    tree = ast.parse(SOURCE.read_text())
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_PrepBase')
    return next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == 'get_params')


def test_public_prep_getter_is_one_existing_native_call():
    body = getter().body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert ast.unparse(body[0].value) == '_PARAMS_METADATA(self, False, _PARAMS_METADATA_CONTEXT)'
    schema = (ROOT / 'python/mojolearn/_native_params_metadata.py').read_text()
    assert "_expansion_prep._PrepBase.get_params: ('declared', False)" in schema


def test_all_inherited_public_getters_have_runtime_probes():
    tree = ast.parse(SOURCE.read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    public = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets))
    def owner(name):
        cls = classes.get(name)
        if cls is None:
            return None
        if any(isinstance(node, ast.FunctionDef) and node.name == 'get_params' for node in cls.body):
            return name
        return next((found for base in cls.bases if isinstance(base, ast.Name)
                     and (found := owner(base.id)) is not None), None)
    selected = {name for name in public if owner(name) == '_PrepBase'}
    tests = ast.parse(RUNTIME_TEST.read_text())
    covered = next(ast.literal_eval(node.value) for node in tests.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'CLASSES' for target in node.targets))
    assert selected == set(covered) and len(selected) == 24
    for name in selected:
        parameters = next(ast.literal_eval(node.value) for node in classes[name].body
                          if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                          and target.id == '_parameters' for target in node.targets))
        assert isinstance(parameters, tuple) and all(type(value) is str for value in parameters)


def test_ignored_deep_value_is_not_evaluated_by_the_wrapper():
    calls = []
    model, context, result = object(), object(), object()
    def entry(*args):
        calls.append(args)
        return result
    class Trap:
        def __bool__(self):
            raise AssertionError('deep has always been ignored by prep getters')
    namespace = dict(_PARAMS_METADATA=entry, _PARAMS_METADATA_CONTEXT=context)
    exec(compile(ast.Module(body=[getter()], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    assert namespace['get_params'](model, Trap()) is result
    assert calls == [(model, False, context)]
