# SPDX-License-Identifier: Apache-2.0
"""Source/schema integrity; installed native behavior has separate probes."""
import ast
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NativeParamsConnection(unittest.TestCase):
    def test_getter_is_one_connected_native_call(self):
        tree = ast.parse((ROOT / 'python/mojolearn/_mode.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ParamsMixin')
        getter = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_params')
        self.assertEqual(len(getter.body), 1)
        self.assertIsInstance(getter.body[0], ast.Return)
        self.assertEqual(ast.unparse(getter.body[0].value.func), '_PARAMS_METADATA')
        registration = (ROOT / 'bindings/_mojolearn_tokenizer_host.mojo').read_text()
        self.assertIn('module.def_py_c_function(params_metadata_c, "params_metadata_native_call")', registration)

    def test_every_declared_builtin_getter_has_an_explicit_schema(self):
        tree = ast.parse((ROOT / 'python/mojolearn/_native_params_metadata.py').read_text())
        schema = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'schemas' for t in n.targets))
        covered = {ast.unparse(key) for key in schema.keys}
        actual = set()
        for path in (ROOT / 'python/mojolearn').glob('*.py'):
            module = ast.parse(path.read_text())
            for cls in module.body:
                if isinstance(cls, ast.ClassDef):
                    for node in cls.body:
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_params':
                            actual.add(path.stem + '.' + cls.name + '.get_params')
        self.assertEqual(covered, actual)

    def test_import_connection_uses_signatures_without_calling_getters(self):
        # Build only the declared schema owners in a fake import package. No
        # product module import, native operation, or constructor is executed.
        text = (ROOT / 'python/mojolearn/_native_params_metadata.py').read_text()
        tree = ast.parse(text)
        schema = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'schemas' for t in n.targets))
        package = ModuleType('_params_connection_fixture')
        package.__path__ = []
        modules = {package.__name__: package}
        for key in schema.keys:
            module_name, class_name, _ = ast.unparse(key).split('.')
            if not hasattr(package, module_name):
                module = ModuleType(package.__name__ + '.' + module_name)
                setattr(package, module_name, module)
                modules[module.__name__] = module
            def make_owner(name):
                def get_params(self, deep=True):
                    raise AssertionError('Python parameter callback ran')
                def __init__(self, alpha=None, *, estimator=None):
                    raise AssertionError('Python constructor ran')
                return type(name, (), {'get_params': get_params, '__init__': __init__,
                                      '__module__': package.__name__ + '.' + module_name,
                                      '_parameters': ('alpha', 'estimator'),
                                      '_parameter_names': ('alpha', 'estimator')})
            setattr(getattr(package, module_name), class_name, make_owner(class_name))
        mode = package._mode
        mode.ParamsMixin._get_param_names = classmethod(lambda cls: (_ for _ in ()).throw(AssertionError('Python signature hook ran')))
        mode.NumericModeMixin = type('NumericModeMixin', (mode.ParamsMixin,), {'__module__': package.__name__})
        package.Model = type('Model', (mode.NumericModeMixin,), {'__module__': package.__name__})
        package.Custom = type('Custom', (mode.NumericModeMixin,), {
            '__module__': package.__name__,
            '_get_param_names': classmethod(lambda cls: (_ for _ in ()).throw(AssertionError('Python signature hook ran')))})
        package.External = type('External', (), {'get_params': lambda self: {}})
        package._training_impl.SGD = type('SGD', (package._training_impl._Optimizer,), {'__module__': package.__name__})
        selection = package.model_selection
        selection._NATIVE_CONTEXT = {'family_msel_search_layouts': {selection._BaseSearch: {'parameters': ('estimator',)}}}
        namespace = {'__name__': package.__name__ + '._native_params_metadata', '__package__': package.__name__}
        exec(compile(text, 'connector', 'exec'), namespace)
        context = {'plans': {}, 'missing': object(), 'pending': {}}
        with patch.dict(sys.modules, modules):
            namespace['connect_classes'](context, package)
        plan = context['plans'][package.Model]
        self.assertEqual(plan['names'], ('alpha', 'estimator', 'numeric_mode'))
        self.assertTrue(plan['recurse'])
        self.assertFalse(context['plans'][package.preprocessing.StandardScaler]['recurse'])
        self.assertEqual(set(context['pending']), {package.Custom})
        self.assertNotIn(package.External, context['pending'])
        self.assertIn('custom Python parameter-name callbacks', context['pending'][package.Custom])
        original = context['plans'][package.Model]
        package.Late = type('Late', (mode.NumericModeMixin,), {'__module__': package.__name__})
        package.Model.__init__ = lambda self, changed=None: None
        with patch.dict(sys.modules, modules):
            namespace['connect_classes'](context, package)
        self.assertIn(package.Late, context['plans'])
        self.assertIs(context['plans'][package.Model], original)
        self.assertTrue(context['ready'])

    def test_native_owner_uses_reference_assignment_and_recursion_guard(self):
        source = (ROOT / 'bindings/params_metadata_native.mojo').read_text()
        self.assertIn('"Py_EnterRecursiveCall"', source)
        self.assertIn('"Py_LeaveRecursiveCall"', source)
        self.assertIn('py_setitem(out, name, value)', source)
        self.assertNotIn('Python.import_module', source)
        self.assertNotIn('msel_clone_parameter', source)
        self.assertNotIn('py_call(model.get_params', source)


if __name__ == '__main__':
    unittest.main()
