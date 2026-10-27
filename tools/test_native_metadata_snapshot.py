# SPDX-License-Identifier: Apache-2.0
"""Source guards for known metadata callbacks; not compiled runtime evidence."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(path, name):
    source = (ROOT / path).read_text()
    return re.search(rf"^def {name}\(.*?(?=^def |^struct |\Z)", source,
                     flags=re.M | re.S).group()


class NativeMetadataSnapshotSourceTests(unittest.TestCase):
    def test_stacking_member_loop_reuses_metadata(self):
        source = function('bindings/trees_public_meta.mojo', '_stacking_fit')
        folds = source[source.index('    for f in range(len(splits)):'):]
        members = folds[folds.index('        for i in range(len(active.ests)):'):
                        folds.index('    if passthrough:')]
        # Regression guard for the demonstrated callbacks/repeated field reads,
        # not a claim that arbitrary PythonObject calls are interpreted Python.
        for pattern in (r'len\((y_full|tr|te)\)', r'trees_address\((Xa|y_full|sw)\)',
                        r'_attr\(model,', r'_view_of_array\('):
            self.assertNotRegex(members, pattern)
        for name in ('x_view', 'y_view', 'meta_view', 'cols_view', 'native_methods', 'native_widths'):
            self.assertIn('var ' + name + ' =', source[:source.index('    for f in range(len(splits)):')])
        self.assertIn('var tr_view =', folds[:folds.index('        for i in range(len(active.ests)):')])
        self.assertIn('var te_view =', folds[:folds.index('        for i in range(len(active.ests)):')])

    def test_prepared_helpers_use_native_metadata_and_retain_owners(self):
        cases = (
            ('bindings/trees_public_boost.mojo', '_gather_view', ('rows', 'cols')),
            ('bindings/trees_public_boost.mojo', '_gather_codes_view', ('codes', 'rows')),
            ('bindings/trees_public_meta.mojo', '_place_view', ('dst', 'rows', 'blk')),
            ('bindings/trees_public_meta.mojo', '_indicator', ('codes',)),
        )
        for path, name, owners in cases:
            with self.subTest(name=name):
                source = function(path, name)
                self.assertNotRegex(source, r'len\(|trees_address\(|_view_of_array\(')
                after_call = source[source.index('    _ = py_call_typed'):]
                for owner in owners:
                    self.assertIn('    _ = ' + owner + '\n', after_call.replace('  #', '\n#'))

    def test_binary_stacking_output_uses_supplied_shape_and_option(self):
        source = function('bindings/trees_public_meta.mojo', '_stack_out')
        self.assertNotIn('_attr(', source)
        self.assertNotIn('_rows(', source)
        self.assertIn('n: Int, binary: Bool', source)

    def test_admission_reads_shape_and_strides_once(self):
        for name, value in (('_view_of_array', 'value'), ('_view_of_buffer', 'view')):
            with self.subTest(name=name):
                source = function('bindings/trees_public_inputs.mojo', name)
                self.assertEqual(source.count(value + '.shape'), 1)
                self.assertEqual(source.count(value + '.strides'), 1)
                loop = source[source.index('    for k in range(rank):'):]
                self.assertNotIn(value + '.shape', loop)
                self.assertNotIn(value + '.strides', loop)

    def test_known_array_lengths_do_not_reappear_in_glue(self):
        for path, names in (
            ('bindings/trees_public_boost.mojo', ('_gather', '_gather_codes', '_gather_vec', '_acc_cols')),
            ('bindings/trees_public_meta.mojo', ('_place', '_acc_cols_w', '_splits', '_indicator')),
        ):
            for name in names:
                with self.subTest(name=name):
                    self.assertNotRegex(function(path, name), r'len\((rows|cols|codes)\)')


if __name__ == '__main__':
    unittest.main()
