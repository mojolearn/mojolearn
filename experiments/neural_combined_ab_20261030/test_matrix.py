import unittest
from matrix import configurations, route_configurations, parse_witness, validate_witness

class RouteEvidenceTests(unittest.TestCase):
    def test_complete_factorial(self):
        cases = configurations()
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({(x['sampler'], x['kv_reuse'], x['engine']) for x in cases}), 8)

    def test_supported_explicit_route_matrix(self):
        rows=route_configurations()
        self.assertEqual(len(rows),11)
        self.assertTrue(all(x['resident_forward']==0 and x['engine']==1 for x in rows if x['route']=='native_engine'))
        self.assertTrue(all(x['sampler']==1 for x in rows if x['route']=='resident'))

    def test_applicability_is_preserved(self):
        events=parse_witness('MOJOLEARN_GENERATION_ROUTE route=legacy_forward capacity=4 produced=4 retained_kv=1 kv_policy=applied kv_uploads_skipped=3')
        self.assertEqual(events[0]['applicability'], {'kv_policy':'applied','kv_uploads_skipped':'3'})

    def test_observed_route_can_shadow_requested_flags(self):
        events = parse_witness('MOJOLEARN_GENERATION_ROUTE route=resident_sampling capacity=4 produced=4 retained_kv=0')
        self.assertEqual(validate_witness(events, 4, 4), ['resident_sampling'])
        self.assertFalse(events[0]['retained_kv'])

    def test_missing_or_repeated_reads_are_rejected(self):
        event = dict(route='legacy_forward', capacity=1, produced=1, retained_kv=False)
        with self.assertRaises(ValueError): validate_witness([event], 4, 1)
        with self.assertRaises(ValueError): validate_witness([event] * 4, 4, 1)

    def test_read_one_witness_consumes_all_tokens(self):
        events = [dict(route='native_engine', capacity=1, produced=i, retained_kv=False) for i in range(1, 5)]
        self.assertEqual(validate_witness(events, 4, 1), ['native_engine'])

if __name__ == '__main__': unittest.main()
