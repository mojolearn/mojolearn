"""Offline acceptance checks; deliberately omit any GPU or fit dependencies."""
import copy
import unittest

from check_neural_startup_readbacks import check


class StartupReceiptChecks(unittest.TestCase):
    def setUp(self):
        record = {'__record__': True, 'library': 'test-stub',
                  'implementation': {'module': 'test', 'name': 'operation'},
                  'params': {'seed': 7, 'operation': 'test-stub'}}
        arm = {'status': 'ok', 'params_record': record, 'ms': [], 'mem': [],
               'digests': [], 'warmup_ms': None, 'info': {'pre_clock_fit': False}}
        self.expected = ['ours', 'opponent']
        self.receipt = {'params_only': True,
                        'arms': {name: copy.deepcopy(arm) for name in self.expected},
                        'params_check': {'verdict': 'MATCHED', 'problems': [],
                            'arms': {name: {'raw_params': copy.deepcopy(record['params'])}
                                     for name in self.expected}}}

    def test_complete_startup(self):
        self.assertEqual(check(self.receipt, self.expected), [])

    def test_own_only_nested_match_cannot_qualify_pair(self):
        del self.receipt['params_check']['arms']['opponent']
        self.receipt['arms']['opponent'] = {'status': 'not_ready'}
        self.assertTrue(check(self.receipt, self.expected))

    def test_deleted_failed_opponent_cannot_qualify_pair(self):
        del self.receipt['arms']['opponent']
        self.assertTrue(check(self.receipt, self.expected))

    def test_round_or_warmup_evidence_rejected(self):
        for key, value in [('ms', [1]), ('warmup_ms', 1), ('digests', ['abc'])]:
            with self.subTest(key=key):
                receipt = copy.deepcopy(self.receipt)
                receipt['arms']['opponent'][key] = value
                self.assertTrue(check(receipt, self.expected))

    def test_readback_disagrees_with_checked_parameters(self):
        self.receipt['arms']['opponent']['params_record']['params']['operation'] = 'other'
        self.assertTrue(check(self.receipt, self.expected))

    def test_constructor_fit_not_accepted(self):
        self.receipt['arms']['opponent']['info']['pre_clock_fit'] = True
        self.assertTrue(check(self.receipt, self.expected))

    def test_missing_implementation_rejected(self):
        self.receipt['arms']['opponent']['params_record']['implementation'] = None
        self.assertTrue(check(self.receipt, self.expected))


if __name__ == '__main__':
    unittest.main()
