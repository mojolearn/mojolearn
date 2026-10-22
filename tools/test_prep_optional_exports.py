"""Exercise prep optional-export probing without loading native bindings.

The arena program runner itself (`_Prog.run`: entry routing, the mandatory
x_prep_run refusal, in-out ranges that skip resident caching) is native since
lane migrate-prog-staging (bindings/arena_prog_native.mojo); its cases run
against the compiled binding, not against a Python stand-in."""
import ast
from pathlib import Path


def _probe():
    path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_expansion_prep.py'
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_optional_prep_entry']
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), scope)
    return scope['_optional_prep_entry']


def test_missing_optional_host_marker_does_not_refuse_gpu():
    class Binding:
        x_prep_run = staticmethod(lambda *args: None)
    assert _probe()(Binding(), 'x_prep_host_column') is None
