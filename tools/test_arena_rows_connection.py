"""Import-time native descriptor admission; no compiled product claim."""
from pathlib import Path
import sys
from types import MemberDescriptorType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]


class Array:
    __slots__ = ('_store', '_base', '_pin', '_mv', '_addr', '_readonly', 'shape',
                 'dtype', 'ndim', 'size', 'nbytes', 'itemsize', 'strides', 'order')


class Rows:
    pass


def connect():
    package = ModuleType('mojolearn')
    package.__path__ = [str(ROOT / 'python/mojolearn')]
    arena = ModuleType('mojolearn._arena_io')
    arena.DeviceRows = Rows
    arrays = ModuleType('mojolearn._array')
    arrays.Array = Array
    bridge = ModuleType('mojolearn._native_classical_bridge')
    bridge._numpy_free_context = lambda: {'array_class': Array}
    tokenizer = ModuleType('mojolearn.tokenizer')
    tokenizer._binding = lambda: SimpleNamespace(_module=SimpleNamespace(arena_rows_native_call=len))
    modules = {'mojolearn': package, 'mojolearn._arena_io': arena,
               'mojolearn._array': arrays, 'mojolearn._native_classical_bridge': bridge,
               'mojolearn.tokenizer': tokenizer}
    with patch.dict(sys.modules, modules):
        namespace = {'__name__': 'mojolearn._native_arena_rows', '__package__': 'mojolearn'}
        source = ROOT / 'python/mojolearn/_native_arena_rows.py'
        exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
        namespace['connect']()
    return arena._ROWS_CONTEXT


class NativeWeakrefConnectionTests(unittest.TestCase):
    def test_real_native_callback_member_is_admitted(self):
        descriptor = vars(weakref.ReferenceType)['__callback__']
        self.assertIs(type(descriptor), MemberDescriptorType)
        context = connect()
        key = 'owner_weakref_callback' if 'owner_weakref_callback' in context else 'arena_weakref_callback'
        self.assertIs(context[key], descriptor)

    def test_python_callback_property_is_refused_without_invocation(self):
        calls = []
        class Foreign:
            __callback__ = property(lambda unused: calls.append('property'))
        with patch.object(weakref, 'ReferenceType', Foreign):
            with self.assertRaisesRegex(TypeError, 'native weak-reference callback descriptor'):
                connect()
        self.assertEqual(calls, [])

    def test_foreign_native_member_is_refused(self):
        class Foreign:
            __callback__ = Array.shape
        with patch.object(weakref, 'ReferenceType', Foreign):
            with self.assertRaisesRegex(TypeError, 'native weak-reference callback descriptor'):
                connect()

    def test_python_weakref_inspection_is_refused_without_invocation(self):
        calls = []
        def foreign(*args):
            calls.append('getweakrefs')
        with patch.object(weakref, 'getweakrefs', foreign):
            with self.assertRaisesRegex(TypeError, 'native weak-reference inspection'):
                connect()
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
