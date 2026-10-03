"""Exercise prep arena routing without loading native bindings."""
import array
import ast
import ctypes
import mmap
import os
from pathlib import Path
import runpy
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def program():
    path = Path(__file__).resolve().parents[1] / 'python/mojolearn/_expansion_prep.py'
    tree = ast.parse(path.read_text())
    constants = {'_MAP_MIN_WORDS', '_R3_NAMES', '_R3_DEFAULT'}
    classes = [n for n in tree.body if (isinstance(n, ast.ClassDef) and n.name in ('_Prog', '_Scratch'))
               or (isinstance(n, ast.FunctionDef) and n.name in
                   ('_optional_prep_entry', '_native_folds', '_zero_words', '_r3'))
               or (isinstance(n, ast.Assign) and any(
                   isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    arena = SimpleNamespace(**runpy.run_path(str(path.with_name('_arena_io.py'))))
    scope = dict(array=array, ctypes=ctypes, os=os, mmap=mmap, time=time, _arena_io=arena)
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


@pytest.mark.parametrize('codes', [None, [0, 1, 0, 1]])
def test_missing_native_folds_is_not_hidden(program, codes):
    """The fold entries are mandatory on both columns (lane cgr4-py-compute
    deleted the Python fold route): a binding without them raises."""
    class Binding:
        x_prep_run = staticmethod(lambda *args: None)
        def __getattr__(self, name):
            raise ImportError('host has no ' + name)
    program['_prep_binding'] = lambda mode: Binding()
    program['_mode'] = lambda: 'identical'
    with pytest.raises(ImportError, match='host has no x_prep_'):
        program['_native_folds'](4, 2, 3, True, codes, 2)


def test_missing_optional_host_marker_does_not_refuse_gpu(program):
    class Binding:
        x_prep_run = staticmethod(lambda *args: None)
    assert program['_optional_prep_entry'](Binding(), 'x_prep_host_column') is None


def test_ranges_return_mutated_input_words_without_resident_caching(program, monkeypatch):
    """LDA metadata and iterative imputation mutate inputs that callers read."""
    class Input:
        dtype = '<f4'
        size = 2

        def __init__(self):
            self.words = array.array('f', [1.0, 2.0])

        def _has_order(self, order):
            return order == 'C'

    def unexpected_cache(*args):
        pytest.fail('mutable input must not reuse a resident copy')

    def run_ranges(base, prog, out, sizes, ranges):
        _, _, output_addr, count = ranges
        words = ctypes.cast(base, ctypes.POINTER(ctypes.c_float))
        outputs = ctypes.cast(output_addr, ctypes.POINTER(ctypes.c_int32))
        # Emulate a device write and download only if explicitly requested.
        returned = [i for r in range(count)
                    for i in range(outputs[4 * r], outputs[4 * r + 1])]
        assert 0 in returned and 1 in returned
        words[0], words[1] = 7.0, 8.0

    binding = SimpleNamespace(x_prep_run=lambda *args: pytest.fail('expected ranges'),
                              x_prep_run_ranges=run_ranges)
    monkeypatch.setenv('MOJOLEARN_ARENA_RANGES', '1')
    monkeypatch.setattr(program['_arena_io'], 'active_cache', unexpected_cache)
    program.update(Array=Input, addr_ro=lambda arr, **kw: arr.words.buffer_info()[0],
                   _prep_binding=lambda mode: binding)
    p = program['_Prog']()
    value = Input()
    off = p.put(value, inout=True)
    p.run('identical')
    assert p.values(off, 2) == [7.0, 8.0]
    assert value.words.tolist() == [1.0, 2.0]
