"""Metadata-only tests: no model, compiler, cloud or numerical execution."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from six_lane_qualification import summarize


class QualificationFactsTest(unittest.TestCase):
    def setUp(self):
        self.row = dict(receipt_sha256='receipt1', source_sha='freeze', vendor='amd',
            execution_status='MEASURED_FULL', quality_assessment='PENDING', reason='original reason',
            candidate_vs_baseline=dict(verdict='SAME', unknown=[]),
            new_full_opponent_review=dict(qualified=[], expected_arms=[]),
            remaining_requirements=['Required identity', 'Durable artifact retention; no automatic source/default changes'])
        self.review = dict(source_sha='freeze', rows=[self.row])
        self.proof = dict(source_sha='freeze', all_required_bytes_preserved=True,
            local_capture_hash_match=True, r2_readback_hash_match=True, remote_inventory_stable=True)

    def test_pending_is_complete_timing_but_not_admitted(self):
        result = summarize(self.review, preservation={'amd': self.proof})
        self.assertEqual(result['completed_timing_pairs'], 1)
        self.assertEqual(result['raw_quality_counts'], {'PENDING': 1})
        facts = result['rows']['receipt1']
        self.assertEqual(facts['baseline_quality'], 'SAME')
        self.assertEqual(facts['independent_reference'], 'ROSTER_NOT_RECORDED')
        self.assertEqual(facts['remaining_recorded_requirements'], ['Required identity'])
        self.assertEqual(len(facts['historical_remaining_requirements']), 2)
        self.assertEqual(facts['raw_reason'], 'original reason')
        self.assertFalse(facts['promotion_authorized'])

    def test_partial_or_other_freeze_retention_never_completes(self):
        for change in ({'r2_readback_hash_match': False}, {'source_sha': 'different'}):
            proof = dict(self.proof, **change)
            facts = summarize(self.review, preservation={'amd': proof})['rows']['receipt1']
            self.assertEqual(facts['preservation'], 'NOT_ESTABLISHED')
            self.assertEqual(len(facts['remaining_recorded_requirements']), 2)

    def test_pair_match_does_not_supply_missing_columns(self):
        identity = dict(source_sha='freeze', qualified_full_identity=False, cases=[dict(
            missing_columns=['apple', 'host', 'nvidia-ptx'], nvidia_amd={
                'A': {'complete_declared_state_pair': {'status': 'MATCH'}},
                'B': {'complete_declared_state_pair': {'status': 'MATCH'}}})])
        result = summarize(self.review, identity)
        self.assertEqual(result['nvidia_amd_same_arm_output_and_state'], {'MATCH': 2})
        self.assertEqual(result['missing_identity_columns']['apple'], 1)
        self.assertFalse(result['full_identity_qualified'])
        identity['cases'].append(dict(nvidia_amd={}))
        result = summarize(self.review, identity)
        self.assertEqual(result['identity_cases'], 2)
        self.assertEqual(result['cases_without_nvidia_amd_comparison'], 1)

    def test_unknown_metrics_do_not_claim_baseline_nonregression(self):
        review = copy.deepcopy(self.review)
        review['rows'][0]['candidate_vs_baseline']['unknown'] = ['new_metric']
        self.assertEqual(summarize(review)['baseline_quality'], {'UNKNOWN': 1})

    def test_duplicate_receipts_and_mixed_freezes_rejected(self):
        with self.assertRaises(ValueError):
            summarize(dict(self.review, rows=[self.row, self.row]))
        with self.assertRaises(ValueError):
            summarize(self.review, {'source_sha': 'other'})
        with self.assertRaises(ValueError):
            summarize(dict(self.review, source_sha='other'))


if __name__ == '__main__':
    unittest.main()
