# SPDX-License-Identifier: Apache-2.0
"""Evidence admission tests; these do not execute a GPU or time CPU work."""
import unittest
from identical_speed_ab import CHECKS, SCREENS, parse_record, compare_pair, selected, profiles


class EvidenceTests(unittest.TestCase):
    def record(self, **changes):
        case = CHECKS[0]
        values = dict(mode="identical", column="nvidia", mask=0,
                      **{k: v for k, v in case.items() if k != "name"},
                      hash=123, ns=0, workspace_bytes=4, slack=4, rpt=8, cpt=4)
        values.update(changes)
        return "SCHEDULE_DEVICE test GPU\nSCHEDULE_DISPATCH test plan\nSCHEDULE_RESULT " + " ".join(f"{k}={v}" for k, v in values.items())

    def test_valid_record(self):
        self.assertEqual(parse_record(self.record(), CHECKS[0], 0, "nvidia")["hash"], 123)

    def test_wrong_mode_vendor_arm_or_fixture(self):
        for change in (dict(mode="fast"), dict(column="amd"), dict(mask=2), dict(k=128)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_record(self.record(**change), CHECKS[0], 0, "nvidia")

    def test_missing_duplicate_results(self):
        for log in ("", self.record() + "\n" + self.record()):
            with self.subTest(log=log[:30]), self.assertRaises(ValueError):
                parse_record(log, CHECKS[0], 0, "nvidia")

    def test_identity_run_cannot_report_a_timing(self):
        with self.assertRaises(ValueError):
            parse_record(self.record(ns=1), CHECKS[0], 0, "nvidia")

    def test_timing_requires_positive_synchronized_duration(self):
        case = SCREENS[0]
        log = self.record(**{k: v for k, v in case.items() if k != "name"})
        with self.assertRaises(ValueError):
            parse_record(log, case, 0, "nvidia")

    def test_changed_bits_or_device_fail_pair(self):
        row = parse_record(self.record(), CHECKS[0], 0, "nvidia")
        for change in (dict(hash=124), dict(device="other GPU")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_pair(row, {**row, **change}, False)

    def test_ratio_direction(self):
        row = parse_record(self.record(), CHECKS[0], 0, "nvidia")
        pair = compare_pair({**row, "ns": 200}, {**row, "ns": 100}, True)
        self.assertEqual(pair["b_over_a"], 0.5)

    def test_profiles_and_case_coverage(self):
        self.assertEqual(len(selected(list(profiles()))), 5)
        self.assertEqual({x["op"] for x in CHECKS}, {0, 1, 2})
        self.assertEqual(len(CHECKS), 18)
        self.assertEqual(len(SCREENS), 18)
        self.assertEqual(len({x["name"] for x in CHECKS + SCREENS}), 36)
        with self.assertRaises(ValueError):
            selected(["typo"])

    def test_single_experiment_branch_selection(self):
        for name in profiles():
            self.assertEqual(list(selected([name])), [name])
        with self.assertRaises(ValueError):
            selected(["one-page", "one-page"])


if __name__ == "__main__":
    unittest.main()
