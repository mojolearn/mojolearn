"""Source-only public-door contracts; no product import or native build."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def classes_from(path, names, namespace, keep=None):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if keep is not None:
        for node in nodes:
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in keep]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


class CompositionDoors(unittest.TestCase):
    def setUp(self):
        self.installed = []
        self.calls = []
        self.result = object()
        context = object()
        self.context = context
        ns = dict(
            install_public_outputs=lambda cls: self.installed.append(('outputs', cls)),
            install_fit_tracking=lambda cls, **kw: self.installed.append(('fits', cls)),
            _application_subclass=lambda cls: False,
            _ENCODE_CONTEXT=context,
            _ENCODE_CALL=lambda *args: self.calls.append(args) or self.result,
        )
        # Execute the actual class-definition hooks with inert installers, then
        # the actual two public class definitions. No mojolearn import occurs.
        classes_from(ROOT / 'python/mojolearn/_mode.py',
                     {'PublicOutputsMixin', 'SklearnTagsMixin'}, ns, {'__init_subclass__'})
        ns['_PrepBase'] = ns['_SelectorMixin'] = ns['SklearnTagsMixin']
        classes_from(ROOT / 'python/mojolearn/_expansion_prep.py', {'RFE', 'IterativeImputer'}, ns)
        self.models = dict(RFE=ns['RFE'](object()), IterativeImputer=ns['IterativeImputer']())
        self.owner_classes = {type(model) for model in self.models.values()}

    def test_actual_definition_hooks_do_not_wrap_composition_classes(self):
        self.assertFalse([row for row in self.installed if row[1] in self.owner_classes])

    def test_public_calls_preserve_arguments_and_return_identity(self):
        x, y, weights = object(), object(), object()
        cases = [
            ('RFE', 'fit', (x, y), {'sample_weight': weights}),
            ('RFE', 'fit_transform', (x, y), {'sample_weight': weights}),
            ('RFE', 'transform', (x,), {}), ('RFE', 'predict', (x,), {}),
            ('RFE', 'predict_proba', (x,), {}), ('RFE', 'decision_function', (x,), {}),
            ('RFE', 'score', (x, y), {}), ('RFE', 'get_support', (True,), {}),
            ('IterativeImputer', 'fit', (x, y), {}),
            ('IterativeImputer', 'fit_transform', (x, y), {}),
            ('IterativeImputer', 'transform', (x,), {}),
        ]
        for owner, operation, args, kwargs in cases:
            with self.subTest(owner=owner, operation=operation):
                self.calls.clear()
                model = self.models[owner]
                self.assertIs(getattr(model, operation)(*args, **kwargs), self.result)
                self.assertEqual(len(self.calls), 1)
                actual_model, actual_operation, actual_args, actual_context = self.calls[0]
                self.assertIs(actual_model, model)
                self.assertEqual(actual_operation, operation)
                self.assertIs(actual_context, self.context)
                self.assertEqual(actual_args[:len(args)], args)
                if kwargs:
                    self.assertIs(actual_args[-1]['sample_weight'], weights)

    def test_uninitialized_inner_context_connects_before_keyword_admission(self):
        source = (ROOT / 'bindings/msel_estimator_native.mojo').read_text()
        section = source.split('if keywords is not None:', 1)[1].split('var values = Python.list()', 1)[0]
        self.assertLess(section.index('classical_connect_numpy(context)'),
                        section.index('classical_admit_value(keywords[keys[k]], context)'))
        self.assertIn('py_type(keywords) is not b.dict', section)
        self.assertIn('py_type(keys[k]) is not b.str', section)

    def test_lifecycle_admits_before_conversion_and_dispatch(self):
        source = (ROOT / 'bindings/prep_composition_lifecycle.mojo').read_text()
        before = source.split('def composition_before(', 1)[1]
        self.assertLess(before.index('composition_value(args[2][keys[i]], context)'),
                        before.index('classical_check_x(args[0]'))
        self.assertIn('prep_estimator_admit(value,', source)
        self.assertNotIn('model.fit(', source)
        owner = (ROOT / 'bindings/prep_owner_native.mojo').read_text()
        for kind, entry in [('rfe', 'prep_rfe_call'), ('iterative_imputer', 'prep_iterimp_call')]:
            section = owner.split('elif kind == "' + kind + '":', 1)[1].split('elif ', 1)[0]
            self.assertLess(section.index('composition_before('), section.index(entry + '('))
        self.assertIn('composition_after_fit(model, args[0], context)', owner)
        self.assertIn('parallel_public_result(result, operation, PythonObject(None), context)', owner)


if __name__ == '__main__':
    unittest.main()
