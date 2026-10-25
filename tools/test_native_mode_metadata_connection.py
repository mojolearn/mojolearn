# SPDX-License-Identifier: Apache-2.0
"""Source connection integrity; native behavior needs the installed runtime tests."""
import ast
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ModeMetadataConnection(unittest.TestCase):
    def test_repaired_entrypoints_are_one_native_call(self):
        tree = ast.parse((ROOT / 'python/mojolearn/_mode.py').read_text())
        methods = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ('is_fitted', '__sklearn_is_fitted__', 'numeric_mode_used', 'vendor_used'):
                methods[node.name] = node
        self.assertEqual(len(methods), 4)
        for node in methods.values():
            body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            self.assertEqual(len(body), 1)
            self.assertIsInstance(body[0], ast.Return)
            self.assertEqual(ast.unparse(body[0].value.func), '_MODE_METADATA')

    def test_registered_route_classes_exist_in_their_source_modules(self):
        tree = ast.parse((ROOT / 'python/mojolearn/_native_mode_metadata.py').read_text())
        known = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'known' for t in n.targets))
        for key in known.keys:
            module, name, method = ast.unparse(key).split('.')
            source = ast.parse((ROOT / 'python/mojolearn' / (module + '.py')).read_text())
            cls = next((n for n in source.body if isinstance(n, ast.ClassDef) and n.name == name), None)
            self.assertIsNotNone(cls, (module, name))
            self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == method for n in cls.body))

    def test_native_entry_is_registered_and_does_not_use_generic_dict_lookup(self):
        native = (ROOT / 'bindings/mode_metadata_native.mojo').read_text()
        registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
        self.assertIn('module.def_py_c_function(mode_metadata_c, "mode_metadata_native_call")', registration)
        self.assertIn('"PyObject_GenericGetDict"', native)
        self.assertNotIn('Python.import_module', native)
        self.assertNotIn('family_raw_get', native)

    def test_connection_preserves_metrics_legacy_getter_and_native_identity(self):
        # Execute only the import-time connector in an isolated fake package;
        # no product modules or binary operations run in this source test.
        package = ModuleType('_mode_connection_fixture')
        package.__path__ = []
        modules = {package.__name__: package}
        for name in ('_backend', '_mode', '_classical_host', '_forest_protocol', '_hierarchy_impl', '_tsa_impl'):
            module = ModuleType(package.__name__ + '.' + name)
            modules[module.__name__] = module
            setattr(package, name, module)
        def owner(name):
            def bind(self, name=None):
                raise AssertionError('instance binding callback must never run')
            return type(name, (), {'_bind': bind})
        package._mode.NumericModeMixin = owner('NumericModeMixin')
        package._forest_protocol.ForestProtocol = owner('ForestProtocol')
        package._hierarchy_impl.AgglomerativeClustering = owner('AgglomerativeClustering')
        package._tsa_impl.ExponentialSmoothing = owner('ExponentialSmoothing')
        package._classical_host._HostBound = owner('_HostBound')
        package._classical_host.HostGaussianProcessClassifier = owner('HostGaussianProcessClassifier')
        package._classical_host._HOST_BASENAMES = {}
        package.Model = type('Model', (package._mode.NumericModeMixin,), {'_BINDING': '_mojolearn_metrics'})
        binary = ModuleType('mojolearn._mojolearn_metrics')
        package._backend.binding = lambda name, mode: binary
        package._backend._vendor_fn = lambda name: 'metrics_vendor'
        namespace = {'__name__': package.__name__ + '._native_mode_metadata', '__package__': package.__name__}
        exec(compile((ROOT / 'python/mojolearn/_native_mode_metadata.py').read_text(), 'connector', 'exec'), namespace)
        with patch.dict(sys.modules, modules):
            for getter in ('metrics_numeric_mode', 'umap_numeric_mode'):
                with self.subTest(getter=getter):
                    setattr(binary, getter, sys.getrecursionlimit)
                    context = {'plans': {}}
                    namespace['connect_classes'](context, package)
                    for mode in ('identical', 'fast'):
                        connection = context['plans'][package.Model]['modes'][mode]
                        self.assertIsNone(connection['error'])
                        self.assertIs(connection['mode'], sys.getrecursionlimit)
                        self.assertEqual(connection['legacy_mode'], 'mojolearn')
                    delattr(binary, getter)
            binary.metrics_numeric_mode = lambda: 1
            context = {'plans': {}}
            namespace['connect_classes'](context, package)
            self.assertIsInstance(context['plans'][package.Model]['modes']['identical']['error'], TypeError)


if __name__ == '__main__':
    unittest.main()
