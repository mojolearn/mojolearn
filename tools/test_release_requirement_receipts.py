import copy
import json
import tempfile
import unittest
from pathlib import Path

import adapter_gate
import autograd_smoke
import gate_autograd_artifacts
import release_requirement_receipts as R


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.report = Path(self.temp.name) / "actual-report.json"
        self.report.write_text('{"producer":"native-runtime","checks":2}\n')
        self.wanted = dict(kind="runtime", source_revision="a" * 40, fingerprint="b" * 64,
                           coverage=["constructors", "callback-refusals"])
        self.receipt = dict(schema=R.SCHEMA, requirement_id="runtime-apple", kind="runtime",
                            source_revision=self.wanted["source_revision"], input_fingerprint=self.wanted["fingerprint"],
                            producer=dict(tool="tools/runtime_probe.py", report_type="execution"),
                            status="passed", exit_code=0,
                            coverage=[dict(id=name, status="passed", exit_code=0, expected=2,
                                           completed=2, failed=0, skipped=0) for name in self.wanted["coverage"]],
                            evidence=[dict(path=str(self.report), sha256=R.sha256(self.report))])

    def errors(self, receipt=None, wanted=None, **kwargs):
        return R.validate_receipt("runtime-apple", wanted or self.wanted, receipt or self.receipt, **kwargs)

    def test_complete_execution_receipt_and_actual_counts(self):
        self.assertEqual(self.errors(), [])
        self.assertEqual(R.coverage_summary(self.receipt), dict(expected=4, completed=4, failed=0, skipped=0))

    def test_plain_log_and_source_report_never_qualify_runtime(self):
        self.assertTrue(self.errors({"status": "passed", "expected": 1, "completed": 1}))
        self.receipt["producer"]["report_type"] = "source"
        self.assertTrue(any("source-only" in error for error in self.errors()))

    def test_source_audit_only_qualifies_its_source_requirement(self):
        self.wanted["kind"] = self.receipt["kind"] = "source"
        self.receipt["producer"]["report_type"] = "source"
        self.assertEqual(self.errors(), [])

    def test_exact_requirement_source_kind_and_input_closure(self):
        for field in ("requirement_id", "kind", "source_revision", "input_fingerprint"):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt[field] = "different"
                self.assertTrue(self.errors(receipt))

    def test_later_commit_does_not_relabel_old_receipt(self):
        later = dict(self.wanted, source_revision="c" * 40)
        self.assertTrue(self.errors(wanted=later))
        # Reuse validates the old receipt against its retained begun manifest.
        self.assertEqual(self.errors(wanted=self.wanted), [])

    def test_required_ids_cannot_be_replaced_with_aggregate_counts(self):
        for action in ("missing", "duplicate", "unknown"):
            with self.subTest(action=action):
                receipt = copy.deepcopy(self.receipt)
                if action == "missing":
                    receipt["coverage"].pop()
                elif action == "duplicate":
                    receipt["coverage"].append(dict(receipt["coverage"][0]))
                else:
                    receipt["coverage"][0]["id"] = "unrelated source check"
                self.assertTrue(self.errors(receipt))

    def test_bad_or_incomplete_coverage_cannot_pass(self):
        for change in (dict(expected=0, completed=0), dict(completed=1), dict(failed=1), dict(skipped=1),
                       dict(expected=True), dict(completed=3), dict(exit_code=True), dict(exit_code=1),
                       dict(status="failed")):
            with self.subTest(change=change):
                receipt = copy.deepcopy(self.receipt)
                receipt["coverage"][0].update(change)
                self.assertTrue(self.errors(receipt))

    def test_failed_attempt_remains_recordable_but_never_passes(self):
        self.receipt.update(status="failed", exit_code=1)
        self.receipt["coverage"][0].update(status="failed", exit_code=1, completed=1, failed=1)
        self.assertEqual(self.errors(require_pass=False), [])
        self.assertTrue(self.errors())

    def test_missing_or_modified_producer_evidence_is_refused(self):
        self.report.write_text("changed bytes")
        self.assertTrue(self.errors())
        self.report.unlink()
        self.assertTrue(self.errors())


class CandidateReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gate-smoke.json"
        self.identity = dict(source="candidate", vendor="nvidia", version="0.8.57", source_commit="a" * 40,
                             release_modes=["identical"], wheels={"mojolearn-core": "c" * 64, "mojolearn-nvidia": "d" * 64})
        self.wanted = dict(kind="candidate", source_revision="a" * 40, fingerprint="b" * 64,
                           coverage=list(R.CANDIDATE_COVERAGE))
        artifacts = dict(schema=gate_autograd_artifacts.SCHEMA, state="built", vendor="nvidia", source_commit="a" * 40,
                         files=[dict(file=name, source=source, exit_code=0, sha256="e" * 64, source_sha256="f" * 64)
                                for name, source in gate_autograd_artifacts.expected("nvidia").items()])
        self.gate = dict(input_mode="candidate", candidate=self.identity, vendor="nvidia", version="0.8.57",
                         source_commit="9" * 40, verdict="PASS", wheel_manifest="setup/wheel-manifest.json", python="venv/bin/python",
                         smoke_rc=0, smoke_cpu_fallback=True, counts={"PASSED": 2, "STARTED": 3}, smoke_lines=5,
                         autograd_rc=0, autograd_counts={"ok": len(autograd_smoke.ENTRIES)}, autograd_artifacts=artifacts,
                         adapters_enabled=True, adapters=dict(ok=True, status="PASSED", rc=0,
                                                              phases={name: 0 for name in adapter_gate.PHASES}),
                         measure_rc=0, measure=dict(planned_whole_plan=7, expected=7, statuses={"process finished": 7},
                                                   pending=0, failed=[], own_not_built=[], board="board.json",
                                                   pairs=dict(matched=9, unchecked=[], refused=[])))
        self.gate.update({name + "_finished": 1.0 for name in ("setup", "smoke", "autograd", "adapters", "measure")})

    def admit(self, gate=None, provenance_change=None):
        self.path.write_text(json.dumps(gate or self.gate))
        proof = dict(schema=R.PROVENANCE_SCHEMA, report_sha256=R.sha256(self.path), requirement_id="candidate-nvidia",
                     kind="candidate", source_revision="a" * 40, input_fingerprint="b" * 64)
        proof.update(provenance_change or {})
        return R.candidate_gate_receipt("candidate-nvidia", self.wanted, self.path, self.identity, proof)

    def test_native_gate_counts_and_tooling_source_are_preserved(self):
        receipt = self.admit()
        self.assertEqual(R.validate_receipt("candidate-nvidia", self.wanted, receipt), [])
        rows = {row["id"]: row for row in receipt["coverage"]}
        self.assertEqual(rows["install"]["completed"], 2)
        self.assertEqual(rows["autograd-artifacts"]["completed"], len(autograd_smoke.ENTRIES))
        self.assertEqual(rows["adapters"]["completed"], len(adapter_gate.PHASES))
        self.assertEqual(rows["measured-plan-pairs"]["completed"], 7)
        self.assertEqual(receipt["measured_pairs"], 9)
        self.assertEqual(receipt["autograd_artifacts"], 8)
        self.assertNotEqual(R.coverage_summary(receipt)["completed"], 5)

    def test_pass_label_without_native_schema_is_refused(self):
        self.assertTrue(R.candidate_gate_errors({"verdict": "PASS"}, self.identity))

    def test_missing_or_wrong_input_provenance_cannot_admit_old_pass(self):
        self.path.write_text(json.dumps(self.gate))
        with self.assertRaisesRegex(ValueError, "hash-bound"):
            R.candidate_gate_receipt("candidate-nvidia", self.wanted, self.path, self.identity, None)
        for field in ("source_revision", "input_fingerprint", "requirement_id", "report_sha256", "kind"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit(provenance_change={field: "wrong"})

    def test_incomplete_native_stages_are_refused_even_with_pass_label(self):
        changes = (
            ("counts", {"FAILED": 1}), ("smoke_lines", 0), ("smoke_rc", None), ("setup_finished", None),
            ("autograd_counts", {"ok": 1}), ("autograd_counts", {"STARTED": len(autograd_smoke.ENTRIES)}),
            ("adapters_enabled", False), ("measure_rc", 1),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                gate = copy.deepcopy(self.gate)
                gate[key] = value
                self.assertTrue(R.candidate_gate_errors(gate, self.identity))

    def test_missing_actual_measurements_pairs_and_adapter_phases_are_refused(self):
        cases = ("partial workloads", "missing pairs", "missing phase", "wrong source", "missing binary", "other wheels")
        for case in cases:
            with self.subTest(case=case):
                gate = copy.deepcopy(self.gate)
                if case == "partial workloads":
                    gate["measure"]["statuses"] = {"process finished": 6}
                elif case == "missing pairs":
                    gate["measure"]["pairs"]["unchecked"] = [dict(race="missing", arm="torch", reason="no evidence")]
                elif case == "missing phase":
                    gate["adapters"]["phases"].pop("max-training")
                elif case == "wrong source":
                    gate["autograd_artifacts"]["source_commit"] = "e" * 40
                elif case == "missing binary":
                    gate["autograd_artifacts"]["files"].pop()
                else:
                    gate["candidate"]["wheels"]["mojolearn-core"] = "f" * 64
                with self.assertRaises(ValueError):
                    self.admit(gate)


if __name__ == "__main__":
    unittest.main()
