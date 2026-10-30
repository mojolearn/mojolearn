import copy
import unittest
from observation_classification import classify


def fresh_report():
    fixtures = ['base', 'ties', 'hashed', 'wide', 'denormal', 'denormal_ftz', 'odd', 'dupes', 'large']
    parts = ['train', 'infer', 'model', 'batch', 'batchgrad', 'batchscale', 'ragged', 'rlpair', 'stepfull']
    rows = [{'lane': 'new-algorithm', 'fixture': f, 'part': p, 'state': 'OWED',
             'value': '0123456789abcdef', 'local_check': 'passed' if p == 'batch' else None,
             'detail': 'no committed record carries this cell part yet'} for f in fixtures for p in parts]
    return {'cells': rows, 'lanes': ['new-algorithm'], 'fixtures': fixtures,
            'execution': {'completed_cells': 9, 'total_cells': 9, 'crashed_cells': []},
            'lanes_summary': {'attempted': 1, 'not_run': 0},
            'native_provenance_complete': True, 'self_test': {'passed': True}}


class FreshReferenceTests(unittest.TestCase):
    def test_complete_observation_with_no_prior_reference_is_provisional(self):
        report = fresh_report()
        original = copy.deepcopy(report)
        result = classify(report, {'exit_code': 4})
        self.assertTrue(result['observation_complete'])
        self.assertTrue(result['fresh_reference_only'])
        self.assertEqual(result['identity_admission'], 'pending')
        self.assertEqual(result['original_exit_code'], 4)
        self.assertEqual(report, original)

    def test_missing_or_bad_evidence_remains_incomplete(self):
        mutations = [
            lambda d: d['cells'][0].update(value=None),
            lambda d: d['cells'][0].update(value='missing'),
            lambda d: d['cells'][0].update(local_check='failed'),
            lambda d: d['cells'][0].update(detail='worker never produced an observation'),
            lambda d: d['cells'][0].update(state='DIVERGENT'),
            lambda d: d['cells'].pop(),
            lambda d: d['cells'].__setitem__(0, copy.deepcopy(d['cells'][1])),
            lambda d: d['execution'].update(crashed_cells=['base']),
            lambda d: d['execution'].update(completed_cells=8),
            lambda d: d.update(native_provenance_complete=False),
            lambda d: d['self_test'].update(passed=False),
            lambda d: d['lanes_summary'].update(not_run=1),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                report = fresh_report()
                mutate(report)
                self.assertFalse(classify(report, {'exit_code': 4})['observation_complete'])

    def test_unrelated_exit_is_not_excused(self):
        for code in (-11, 1, 2, 3, 124):
            with self.subTest(code=code):
                self.assertFalse(classify(fresh_report(), {'exit_code': code})['observation_complete'])


if __name__ == '__main__':
    unittest.main()
