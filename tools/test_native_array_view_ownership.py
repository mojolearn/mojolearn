# SPDX-License-Identifier: Apache-2.0
"""Source connection and ownership invariants; native lifetime probes are separate."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeArrayViewOwnership(unittest.TestCase):
    def test_array_input_and_public_outputs_use_same_retained_owner(self):
        source = (ROOT / 'bindings/sequential_inputs.mojo').read_text()
        body = source.split('def sequential_input(', 1)[1].split('\ndef ', 1)[0]
        self.assertIn('return native_array_ndarray_view(value, context)', body)
        self.assertNotIn('array_fields', body)
        public = (ROOT / 'bindings/public_attributes_native.mojo').read_text()
        self.assertIn('from bindings.native_array_view import native_array_ndarray_view as _array_view', public)
        self.assertNotIn('def _array_view(', public)

    def test_both_context_builders_pin_complete_native_storage_metadata(self):
        expected = {'_store', '_mv', '_addr', '_readonly', 'shape', 'dtype', 'order'}
        for name in ('_native_classical_bridge.py', '_x_sequence_model.py'):
            tree = ast.parse((ROOT / 'python/mojolearn' / name).read_text())
            declaration = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == 'array_storage_slots' for t in n.targets))
            self.assertEqual(ast.literal_eval(declaration.value.generators[0].iter),
                             ('_store', '_mv', '_addr', '_readonly', 'shape', 'dtype', 'order'))
            self.assertEqual(set(ast.literal_eval(declaration.value.generators[0].iter)), expected)
            self.assertEqual(ast.unparse(declaration.value.value), 'vars(Array)[name]')
            dictionaries = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                            and any(isinstance(k, ast.Constant) and k.value == 'array_storage_slots' for k in n.keys)]
            self.assertEqual(len(dictionaries), 1)
            values = {k.value: ast.unparse(v) for k, v in zip(dictionaries[0].keys, dictionaries[0].values) if isinstance(k, ast.Constant)}
            self.assertEqual(values['array_storage_slots'], 'array_storage_slots')
            self.assertEqual(values['array_storage_dtypes'], 'SUPPORTED_DTYPES')

    def test_retained_owner_and_mapped_slice_checks_precede_export(self):
        source = (ROOT / 'bindings/native_array_view.mojo').read_text()
        self.assertIn('var store = values["_store"]', source)
        self.assertIn('store = values["_mv"]', source)
        self.assertIn('store.obj is None', source)
        self.assertIn('var buffer = native_buffer_view(store)', source)
        self.assertLess(source.index('store.obj is None'), source.index('native_buffer_view(store)'))
        self.assertIn('offset=offset, order=values["order"]', source)
        self.assertIn('py_call(buffer.toreadonly)', source)
        self.assertIn('Int32(0x98)', source)
        for forbidden in ('Python.import_module', '.__array__(', '.from_address(', '.asarray(', 'PyObject_GenericGetAttr'):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
