# SPDX-License-Identifier: Apache-2.0
"""Import-time slot selection; these tests do not claim native runtime coverage."""
import ast
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PublicAttributeConnection(unittest.TestCase):
    def fixture(self):
        package = ModuleType('_attribute_fixture')
        package.__path__ = []
        mode = ModuleType(package.__name__ + '._mode')
        package._mode = mode
        def original(self, name):
            raise AssertionError('product getter ran during connection')
        mode.PublicOutputsMixin = type('PublicOutputsMixin', (), {'__getattribute__': original, '__module__': mode.__name__})
        mode.ParamsMixin = type('ParamsMixin', (mode.PublicOutputsMixin,), {'__module__': mode.__name__})
        mode.NumericModeMixin = type('NumericModeMixin', (), {'__module__': mode.__name__})
        bridge = ModuleType(package.__name__ + '._native_classical_bridge')
        bridge._IMPORT_LAYOUTS = []  # Import-time layout registry used by the extracted helper.
        tree = ast.parse((ROOT / 'python/mojolearn/_native_classical_bridge.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_layout')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'layout', 'exec'), vars(bridge))
        bridge._numpy_free_context = lambda: (_ for _ in ()).throw(AssertionError('fixture context replaced'))
        array = ModuleType(package.__name__ + '._array')
        cholesky = ModuleType(package.__name__ + '._cholesky_impl')
        cholesky.Cholesky = type('Cholesky', (), {'logdet_': property(lambda self: 1)})
        tokenizer = ModuleType(package.__name__ + '.tokenizer')
        tokenizer._binding = lambda: (_ for _ in ()).throw(AssertionError('fixture imported binary'))
        namespace = {'__name__': package.__name__ + '._native_public_attributes', '__package__': package.__name__}
        exec(compile((ROOT / 'python/mojolearn/_native_public_attributes.py').read_text(), 'connector', 'exec'), namespace)
        # Test-only descriptor substitutes for the already-connected native
        # factory product; no fake result is reported as native qualification.
        def descriptor(self, name):
            raise AssertionError('installed getter ran during connection')
        context = dict(plans={}, pending={}, missing=object(), descriptor=descriptor,
                       property_schemas={vars(cholesky.Cholesky)['logdet_']: 'cholesky_logdet'}, stl_connected=False)
        namespace['_CONTEXT'] = context
        modules = {m.__name__: m for m in (package, mode, bridge, tokenizer, cholesky, array)}
        return package, mode, namespace, context, modules, original

    def test_unknown_properties_and_getattr_keep_original_slots(self):
        package, mode, ns, context, modules, original = self.fixture()
        def cls(name, base=mode.ParamsMixin, **attrs):
            value = type(name, (base,), dict(__module__=package.__name__ + '.models', **attrs))
            setattr(package, name, value)
            return value
        base = cls('Base')
        plain = cls('Plain', base)
        prop = cls('Unported', base, state=property(lambda self: (_ for _ in ()).throw(AssertionError('property invoked'))))
        hook = cls('Hook', base, __getattr__=lambda self, name: (_ for _ in ()).throw(AssertionError('hook invoked')))
        with patch.dict(sys.modules, modules):
            ns['connect_classes'](package)
        self.assertIn(base, context['plans'])
        self.assertIn(plain, context['plans'])
        self.assertIs(vars(prop)['__getattribute__'], original)
        self.assertIs(vars(hook)['__getattribute__'], original)
        self.assertIn('unconnected property', context['pending'][prop])
        self.assertIn('Python __getattr__', context['pending'][hook])
        self.assertNotIn(mode.PublicOutputsMixin, context['plans'])
        self.assertIs(vars(mode.PublicOutputsMixin)['__getattribute__'], original)

    def test_lazy_subclass_admission_does_not_refresh_old_plans(self):
        package, mode, ns, context, modules, original = self.fixture()
        package.Base = type('Base', (mode.ParamsMixin,), {'__module__': package.__name__ + '.models'})
        with patch.dict(sys.modules, modules):
            ns['connect_classes'](package)
        old = context['plans'][package.Base]
        package.Late = type('Late', (package.Base,), {'__module__': package.__name__ + '.models'})
        package.Unported = type('Unported', (package.Base,), {'__module__': package.__name__ + '.models', 'value': property(lambda self: 1)})
        with patch.dict(sys.modules, modules):
            ns['connect_classes'](package)
        self.assertIn(package.Late, context['plans'])
        self.assertIs(context['plans'][package.Base], old)
        self.assertIs(vars(package.Unported)['__getattribute__'], original)

    def test_existing_native_slot_owner_is_not_changed(self):
        package, mode, ns, context, modules, original = self.fixture()
        def existing(self, name):
            raise AssertionError('existing getter ran')
        package.Owned = type('Owned', (mode.ParamsMixin,), {'__module__': package.__name__ + '.models', '__getattribute__': existing})
        with patch.dict(sys.modules, modules):
            ns['connect_classes'](package)
        self.assertIs(vars(package.Owned)['__getattribute__'], existing)
        self.assertNotIn(package.Owned, context['plans'])
        self.assertNotIn(package.Owned, context['pending'])

    def test_only_explicit_builtin_property_identity_is_connected(self):
        package, mode, ns, context, modules, _ = self.fixture()
        prop = next(iter(context['property_schemas']))
        package.Cholesky = type('Cholesky', (mode.ParamsMixin,), {'__module__': package.__name__ + '.models', 'logdet_': prop})
        package.Fake = type('Cholesky', (mode.ParamsMixin,), {'__module__': package.__name__ + '.models', 'logdet_': property(lambda self: 1)})
        with patch.dict(sys.modules, modules):
            ns['connect_classes'](package)
        self.assertEqual(context['plans'][package.Cholesky]['properties']['logdet_'], (prop, 'cholesky_logdet'))
        self.assertNotIn(package.Fake, context['plans'])
        self.assertIn('unconnected property', context['pending'][package.Fake])

    def test_only_registered_archive_finalizers_are_connected(self):
        import ctypes
        from types import MethodType
        factory = ctypes.pythonapi.PyInstanceMethod_New
        factory.argtypes = [ctypes.py_object]
        factory.restype = ctypes.py_object
        package, mode, ns, context, modules, original = self.fixture()
        context['descriptor'] = factory(MethodType(len, {}))
        context['archive_finalizer_entry'] = len
        archive = dict(host_archive_lifetime_layouts={})
        finalizer = factory(MethodType(len, archive))
        archive['host_archive_finalizer_descriptor'] = finalizer
        owner = ModuleType(package.__name__ + '._native_archive_bridge')
        owner._CONTEXT = {'host_archive_context': archive}
        modules[owner.__name__] = owner
        def cls(name, member, base=mode.ParamsMixin):
            result = type(name, (base,), {'__module__': package.__name__ + '.models', '__del__': member})
            setattr(package, name, result)
            return result
        registered = cls('Registered', finalizer)
        fields_descriptor = next(vars(base)['__dict__'] for base in registered.__mro__ if '__dict__' in vars(base))
        archive['host_archive_lifetime_layouts'][registered] = (None, fields_descriptor)
        inherited = cls('Inherited', finalizer, registered)
        unregistered = cls('Unregistered', finalizer)
        other_descriptor = cls('OtherDescriptor', factory(MethodType(len, archive)))
        calls = []
        class ForeignDescriptor:
            def __get__(self, instance, owner=None):
                calls.append('hook')
                raise AssertionError('foreign descriptor executed')
        foreign = cls('Foreign', ForeignDescriptor())
        with patch.dict(sys.modules, modules):
            ns['connect_classes'](package)
        for value in (registered, inherited):
            self.assertIn(value, context['plans'])
            self.assertIs(context['plans'][value]['bindings']['__del__'][0], finalizer)
        for value in (unregistered, other_descriptor, foreign):
            self.assertEqual(context['pending'][value], 'unconnected descriptor: __del__')
            self.assertIs(vars(value)['__getattribute__'], original)
        self.assertEqual(calls, [])

    def test_archive_finalizer_context_and_entry_must_match(self):
        import ctypes
        from types import MethodType
        factory = ctypes.pythonapi.PyInstanceMethod_New
        factory.argtypes = [ctypes.py_object]
        factory.restype = ctypes.py_object
        package, mode, ns, context, modules, _ = self.fixture()
        context['descriptor'] = factory(MethodType(len, {}))
        context['archive_finalizer_entry'] = len
        cls = type('Registered', (mode.ParamsMixin,), {'__module__': package.__name__ + '.models'})
        archive = {'host_archive_lifetime_layouts': {cls: ()}}
        for bound in (MethodType(id, archive), MethodType(len, {})):
            with self.subTest(bound=bound.__func__.__name__):
                descriptor = factory(bound)
                archive['host_archive_finalizer_descriptor'] = descriptor
                self.assertIsNone(ns['_archive_finalizer_binding'](cls, descriptor, context, [archive]))

    def test_registration_indent_and_snapshot_order(self):
        registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
        for line in registration.splitlines():
            if 'module.def_py_c_function(' in line:
                self.assertTrue(line.startswith('        '), line)
        source = (ROOT / 'python/mojolearn/__init__.py').read_text()
        self.assertLess(source.index('_connect_public_attributes('), source.index('_connect_native_cv('))
        self.assertLess(source.index('_connect_native_cv('), source.index('_connect_params_metadata_classes('))
        lazy = (ROOT / 'python/mojolearn/_expansion_sequence.py').read_text()
        self.assertLess(lazy.index('connect_attributes(sys.'), lazy.index('connect_classes(_mode.'))

    def test_no_python_getter_or_audited_frame_callback_in_native_owner(self):
        source = (ROOT / 'bindings/public_attributes_native.mojo').read_text()
        for forbidden in ('Python.import_module', 'PyObject_GenericGetAttr', '._getframe(', '.f_code', '.fget(', '.__array__('):
            self.assertNotIn(forbidden, source)
        for required in ('"PyEval_GetFrame"', '"PyFrame_GetCode"', '"PyFrame_GetBack"', '_classical_descriptor_get', '_array_view'):
            self.assertIn(required, source)
        self.assertIn('"PyMethod_New"', source)
        self.assertNotIn('member.__get__', source)


if __name__ == '__main__':
    unittest.main()
