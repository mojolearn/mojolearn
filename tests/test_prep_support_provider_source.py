"""Source-only forwarding control; no compiled/runtime qualification."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_support_shell_passes_original_objects_and_finished_result():
    tree = ast.parse((ROOT / 'python/mojolearn/_expansion_prep.py').read_text())
    selector = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == '_SelectorMixin')
    method = next(node for node in selector.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'get_support')
    calls = []
    owner, flag, context, finished = object(), object(), object(), object()

    def entry(*args):
        calls.append(args)
        return finished

    namespace = {'_STATELESS_CALL': entry, '_STATELESS_CONTEXT': context}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<support-shell>', 'exec'), namespace)
    assert namespace['get_support'](owner, flag) is finished
    assert calls == [(owner, 'get_support', (flag,), context)]


