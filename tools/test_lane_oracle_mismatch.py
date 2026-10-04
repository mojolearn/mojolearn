# SPDX-License-Identifier: Apache-2.0
"""Batched numerical mismatch oracles must be classified at their callsite."""
import ast
import textwrap
import unittest

import lane_applicability as applicability


def oracle(body):
    node = ast.parse('def body(ml, X, yc, yr, Xh=None):\n' + textwrap.indent(textwrap.dedent(body), '    ')).body[0]
    return applicability._oracle(node)[0]


class BatchedOracle(unittest.TestCase):
    def test_raised_explicit_comparisons_are_independent(self):
        self.assertEqual(oracle('''
            a = ml.Thing().fit(X)
            b = ml.Thing().fit(X)
            mismatch = _oracle_mismatch(("a", a.coef_, "b", b.coef_))
            if mismatch:
                raise NumericalMismatch(mismatch, {})
        '''), 'independent')

    def test_starred_comprehension_preserves_compared_objects(self):
        self.assertEqual(oracle('''
            m = ml.RadiusNeighbors().fit(X)
            par = _ragged(_pq(m, X, "radius_neighbors"))
            plain = _ragged(m.radius_neighbors(X))
            mismatch = _oracle_mismatch(*[(name, par[k], name, plain[k])
                                          for k, name in enumerate(("counts", "dist", "idx"))])
            if mismatch:
                raise NumericalMismatch(mismatch, {})
        '''), 'independent')

    def test_shared_object_remains_self_comparison(self):
        self.assertEqual(oracle('''
            a = ml.Thing().fit(X)
            left = a.transform(X)
            right = left[:8]
            mismatch = _oracle_mismatch(*[("left", left, "right", right)])
            if mismatch:
                raise NumericalMismatch(mismatch, {})
        '''), 'self')

    def test_dropped_or_returned_message_is_not_an_oracle(self):
        for ending in ('', 'return mismatch', 'if unrelated:\n    raise NumericalMismatch("unrelated", {})'):
            with self.subTest(ending=ending):
                self.assertEqual(oracle('''
                    a = ml.Thing().fit(X)
                    b = ml.Thing().fit(X)
                    mismatch = _oracle_mismatch(("a", a.coef_, "b", b.coef_))
                ''' + '\n' + textwrap.indent(ending, '                    ')), 'recorded')
        self.assertEqual(oracle('''
            _oracle_mismatch(("a", x, "b", y))
        '''), 'recorded')

    def test_unknown_starred_operands_are_not_assumed_independent(self):
        self.assertEqual(oracle('''
            mismatch = _oracle_mismatch(*dynamic_pairs)
            if mismatch:
                raise NumericalMismatch(mismatch, {})
        '''), 'recorded')

    def test_comprehension_bound_objects_do_not_invent_independence(self):
        self.assertEqual(oracle('''
            mismatch = _oracle_mismatch(*[("a", left, "b", right) for left, right in dynamic_pairs])
            if mismatch:
                raise NumericalMismatch(mismatch, {})
        '''), 'recorded')

    def test_real_radius_lane_remains_multi_device_and_excluded_from_one_gpu(self):
        scope = applicability.scopes()['par-queries-radius']
        self.assertEqual(scope.oracle, 'independent')
        self.assertEqual(scope.claim_devices, 2)
        self.assertFalse(scope.applicable('nvidia-1gpu')[0])
        self.assertTrue(scope.applicable('nvidia-2gpu')[0])


if __name__ == '__main__':
    unittest.main()
