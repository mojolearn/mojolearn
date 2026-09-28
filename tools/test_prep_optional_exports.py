"""Exercise prep arena routing without loading native bindings."""
import array
import ast
import ctypes
import os
from pathlib import Path

import pytest


@pytest.fixture
def program():
    path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_expansion_prep.py'
    tree = ast.parse(path.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('_Prog', '_Scratch')]
    scope = dict(array=array, ctypes=ctypes, os=os)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


@pytest.mark.parametrize('missing', [AttributeError, ImportError])
@pytest.mark.parametrize('exports', [(), ('x_prep_run_scratch',), ('x_prep_run_out',)])
def test_optional_exports_preserve_arena_routes(program, missing, exports):
    calls = []
    class Binding:
        def __getattr__(self, name):
            if name == 'x_prep_run' or name in exports:
                return lambda *args: calls.append((name, args))
            raise missing(name)
    program['_prep_binding'] = lambda mode: Binding()
    p = program['_Prog']()
    p.alloc(3)
    p.scratch(5)
    p.out_size = 7
    p.run('identical')
    expected = ('x_prep_run_out' if 'x_prep_run_out' in exports else
                'x_prep_run_scratch' if 'x_prep_run_scratch' in exports else 'x_prep_run')
    assert [name for name, args in calls] == [expected]
    assert len(p.arena) == (3 if expected == 'x_prep_run_out' else
                            10 if expected == 'x_prep_run_scratch' else 15)


def test_missing_mandatory_entry_is_not_hidden(program):
    class Binding:
        def __getattr__(self, name):
            raise ImportError('host loader failed')
    program['_prep_binding'] = lambda mode: Binding()
    with pytest.raises(ImportError, match='host loader failed'):
        program['_Prog']().run('identical')
