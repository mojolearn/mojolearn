"""Import-time native descriptor admission; no compiled product claim."""
from pathlib import Path
import sys
import threading
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


class Scope:
    pass


class Resident:
    def take(self):
        raise AssertionError("import-time metadata must never call take")


class Cache:
    @staticmethod
    def _key(value):
        raise AssertionError("import metadata must never call key")

    def get(self, value):
        raise AssertionError("import metadata must never call get")


def connect(native_entry=len, store_binding=None, backend_fields=None):
    package = ModuleType('mojolearn')
    package.__path__ = [str(ROOT / 'python/mojolearn')]
    arena = ModuleType('mojolearn._arena_io')
    arena.DeviceRows = Rows
    arena.DeviceCache = Cache
    arena._Scope = Scope
    arena._LOCAL = threading.local()
    selection = ModuleType('mojolearn.model_selection')
    selection._Resident = Resident
    buffers = ModuleType('mojolearn._buffer')
    buffers._native = lambda name: native_entry
    arrays = ModuleType('mojolearn._array')
    arrays.Array = Array
    bridge = ModuleType('mojolearn._native_classical_bridge')
    bridge._numpy_free_context = lambda: {'array_class': Array}
    tokenizer = ModuleType('mojolearn.tokenizer')
    tokenizer._binding = lambda: SimpleNamespace(_module=SimpleNamespace(arena_rows_native_call=len))
    backend = ModuleType('mojolearn._backend')
    backend.binding = lambda name, mode: SimpleNamespace() if store_binding is None else store_binding
    if backend_fields:
        vars(backend).update(backend_fields)
    modules = {'mojolearn': package, 'mojolearn._arena_io': arena,
               'mojolearn._array': arrays, 'mojolearn._native_classical_bridge': bridge,
               'mojolearn.tokenizer': tokenizer, 'mojolearn.model_selection': selection,
               'mojolearn._buffer': buffers, 'mojolearn._backend': backend}
    with patch.dict(sys.modules, modules):
        namespace = {'__name__': 'mojolearn._native_arena_rows', '__package__': 'mojolearn'}
        source = ROOT / 'python/mojolearn/_native_arena_rows.py'
        exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
        namespace['connect']()
    return arena._ROWS_CONTEXT


class NativeWeakrefConnectionTests(unittest.TestCase):
    def test_cache_upload_refuses_preimport_exception_constructor_replacement(self):
        touched = []
        def function(*args):
            touched.append("function")
        class Foreign(RuntimeError):
            def __new__(cls, *args):
                touched.append("new")
                return super().__new__(cls, *args)
        Foreign.__name__ = "RuntimeError"
        Foreign.__module__ = "builtins"
        class Meta(type):
            def __getattribute__(cls, name):
                touched.append(name)
                return super().__getattribute__(name)
        class ForeignMeta(RuntimeError, metaclass=Meta):
            pass
        # Each replacement is installed before the connector module source
        # executes; this checks actual import setup, not a mocked validator.
        for replacement in (function, Foreign, ForeignMeta, ValueError, int):
            with self.subTest(replacement_type=type(replacement)):
                failure = None
                with patch("builtins.RuntimeError", replacement):
                    try:
                        connect()
                    except TypeError as error:
                        failure = str(error)
                # unittest's traceback clearer itself catches RuntimeError;
                # run that observer only after restoring the poisoned name.
                self.assertEqual(failure, "DeviceCache requires the immutable native RuntimeError builtin")
        self.assertEqual(touched, [])

    def test_cache_supports_retains_native_string_and_exception_entries(self):
        context = connect()
        self.assertIs(context["cache_str_add"], str.__add__)
        self.assertIs(context["cache_import_error"], ImportError)
        self.assertIs(context["cache_error_new"], BaseException.__new__)
        self.assertEqual(context["cache_cpu_error_templates"], ())

    def test_cache_supports_tokenizes_cpu_templates_before_inserting_values(self):
        class Host(ModuleType):
            pass
        class Missing(ModuleType):
            pass
        class NoCpuEntry(AttributeError):
            pass
        def message(module, entry, reason, basename=None):
            return f"{module}.{entry}: {reason}; host {basename or module}"
        context = connect(backend_fields=dict(_HostBinding=Host, _NoGpuBinding=Missing,
                                              NoCpuEntry=NoCpuEntry, _no_cpu_implementation=message))
        values = ("mod\0cache-reason\0", "entry\0cache-module\0", "why\0cache-host\0", "host\0cache-entry\0")
        for index, template in enumerate(context["cache_cpu_error_templates"]):
            actual = "".join(part if type(part) is str else values[part] for part in template)
            expected = message(*values[:3], values[3] if index == 0 else None)
            self.assertEqual(actual, expected)

    def test_cache_supports_refuses_preimport_import_error_replacement(self):
        touched = []
        class Foreign(ImportError):
            def __new__(cls, *args):
                touched.append("new")
                return super().__new__(cls, *args)
        Foreign.__name__ = "ImportError"
        Foreign.__module__ = "builtins"
        for replacement in (Foreign, ValueError, lambda *args: touched.append("function")):
            with self.subTest(replacement_type=type(replacement)):
                failure = None
                with patch("builtins.ImportError", replacement):
                    try:
                        connect()
                    except TypeError as error:
                        failure = str(error)
                self.assertEqual(failure, "DeviceCache requires the immutable native ImportError builtin")
        self.assertEqual(touched, [])

    def test_cache_live_retains_connected_native_counter(self):
        binding = SimpleNamespace(x_prep_dev_put=len, x_prep_dev_free=abs,
                                  x_prep_dev_live=hash)
        context = connect(store_binding=binding)
        self.assertEqual(context["cache_live_entries"], (hash, hash))

    def test_cache_live_refuses_import_time_python_counter(self):
        touched = []
        def foreign():
            touched.append("counter")
            return 0
        binding = SimpleNamespace(x_prep_dev_put=len, x_prep_dev_free=abs,
                                  x_prep_dev_live=foreign)
        with self.assertRaisesRegex(TypeError, "compiled native live-slot entry"):
            connect(store_binding=binding)
        self.assertEqual(touched, [])

    def test_cpu_cache_upload_retains_empty_store_registry_and_native_setter(self):
        context = connect()
        self.assertEqual(context['cache_store_routes'], ())
        self.assertEqual(context['cache_live_entries'], ())
        self.assertIs(context['cache_object_setattr'], object.__setattr__)
        self.assertIs(context['cache_runtime_error'], RuntimeError)
        self.assertEqual(context['cache_counter_types'],
                         (type(None), float, complex, str, bytes, bytearray,
                          list, tuple, dict, set, frozenset, range))
        self.assertEqual(context['cache_entry_types'],
                         (bool, int, float, complex, str, bytes, bytearray,
                          list, tuple, dict, set, frozenset, range))

    def test_cache_lookup_retains_key_and_scalar_metadata_without_calling_it(self):
        context = connect()
        self.assertIs(context['cache_key_descriptor'], vars(Cache)['_key'])
        self.assertIs(context['cache_key_function'], Cache._key)
        self.assertEqual(context['cache_integer_types'],
                         (type(None), bool, int, float, complex, str, bytes, bytearray, list, tuple, dict, set, frozenset))
        self.assertIs(context['cache_scalar_context']['modules'], sys.modules)
        self.assertIs(context['cache_deprecation_warning'], DeprecationWarning)
        self.assertEqual(context['cache_primitive_bases'], (int, float, complex, str, bytes, bytearray))
        self.assertEqual(context['cache_type_flags'](Cache), Cache.__flags__)

    def test_cache_statistics_retain_native_attribute_and_thread_local_entries(self):
        context = connect()
        self.assertIs(context['cache_object_getattribute'], object.__getattribute__)
        self.assertIs(context['cache_namespace_class'], SimpleNamespace)
        self.assertIs(context['cache_namespace_getattribute'], SimpleNamespace.__getattribute__)
        self.assertIs(context['cache_member_type'], MemberDescriptorType)
        self.assertIs(context['cache_local_type'], type(threading.local()))
        self.assertIs(context['cache_local_getattribute'], type(threading.local()).__getattribute__)
        self.assertEqual(context['cache_hash_types'], (type(None), bool, int, float, complex, str, bytes))

    def test_owner_release_retains_native_constructor_and_scalar_identities(self):
        context = connect()
        expected = dict(type=type, object=object, list=list, tuple=tuple, dict=dict,
                        str=str, bytes=bytes, bytearray=bytearray, bool=bool,
                        int=int, float=float, complex=complex, memoryview=memoryview)
        self.assertEqual(context['owner_native_types'], expected)
        for name, native_type in expected.items():
            self.assertIs(context['owner_native_types'][name], native_type)

    def test_native_resident_entries_are_connected_without_calling_python_take(self):
        context = connect()
        self.assertIs(context['arena_resident_take'], Resident.take)
        self.assertTrue(all(context[key] is len for key in
                            ('arena_msel_put', 'arena_msel_free', 'arena_msel_take_rows', 'arena_cast')))
        self.assertIs(type(context['arena_half_copy_canonical']), bool)

    def test_python_resident_entry_is_refused_before_invocation(self):
        calls = []
        def forbidden(*args):
            calls.append(True)
        with self.assertRaisesRegex(TypeError, 'compiled resident and cast entries'):
            connect(forbidden)
        self.assertEqual(calls, [])

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
