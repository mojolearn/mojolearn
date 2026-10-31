import copy
import json
import tempfile
import unittest
from pathlib import Path

import adapter_gate
import autograd_smoke
import gate_autograd_artifacts
import release_requirement_receipts as R


class UnsupportedReceiptTests(unittest.TestCase):
    def test_caller_typed_receipt_and_hashed_plaintext_never_prove_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "no-execution.log"
            path.write_text("No build or execution occurred.\n")
            for kind in ("compile", "runtime", "identity", "adapters", "performance", "published"):
                with self.subTest(kind=kind):
                    wanted = dict(kind=kind, source_revision="a" * 40, fingerprint="b" * 64, coverage=["check"])
                    receipt = dict(schema=R.SCHEMA, requirement_id="test", kind=kind,
                        source_revision=wanted["source_revision"], input_fingerprint=wanted["fingerprint"],
                        producer=dict(tool="anything", report_type="build" if kind == "compile" else "execution"),
                        status="passed", exit_code=0,
                        coverage=[dict(id="check", expected=1, completed=1, failed=0, skipped=0, status="passed", exit_code=0)],
                        evidence=[dict(path=str(path), sha256=R.sha256(path))])
                    errors = R.validate_receipt("test", wanted, receipt)
                    self.assertTrue(any("unsupported actual producer" in error for error in errors))
                    receipt["producer"]["tool"] = "tools/gate_smoke_box.py"
                    self.assertTrue(R.validate_receipt("test", wanted, receipt))

    def test_plain_pass_log_has_no_receipt_contract(self):
        self.assertTrue(R.validate_receipt("runtime-apple", dict(kind="runtime"), {"status": "passed"}))


class SourceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "endpoints.json"
        self.inputs = dict(contract=dict(id="runtime-source", kind="source"), configuration=dict(release_modes=["identical"]),
                           product_api={"python/mojolearn/example.py": "a" * 40})
        self.wanted = dict(kind="source", source_revision="a" * 40,
                           fingerprint=R.hashlib.sha256(json.dumps(self.inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                           coverage=list(R.SOURCE_COVERAGE), inputs=self.inputs)
        self.report = dict(schema=2, scope=R.SOURCE_SCOPE,
            endpoints=[dict(endpoint="mojolearn.Model.predict", public="mojolearn.Model", method="predict", kind="method",
                            runtime_status="native_boundary", static_verdict="native", native_entries=["native_predict"],
                            definition="python/mojolearn/example.py:1", source_ast_sha256="a" * 64),
                       dict(endpoint="mojolearn.Model.__sklearn_tags__", public="mojolearn.Model", method="__sklearn_tags__",
                            kind="method", runtime_status="allowed_tag_metadata", static_verdict="python", native_entries=[],
                            definition="python/mojolearn/example.py:3", source_ast_sha256="b" * 64)],
            summary=dict(endpoints=2, by_kind={"method": 2}, by_status={"native_boundary": 1, "allowed_tag_metadata": 1}),
            gate=dict(status="source-classification-passed", prohibited_or_unresolved=0, problems=[], runtime_proof_required=True))

    def admit(self, report=None, proof_change=None):
        self.path.write_text(json.dumps(self.report if report is None else report))
        proof = dict(schema=R.PROVENANCE_SCHEMA, requirement_id="runtime-source", kind="source", run_id="actual-source-run",
                     source_revision=self.wanted["source_revision"], input_fingerprint=self.wanted["fingerprint"], report_sha256=R.sha256(self.path))
        proof.update(proof_change or {})
        (self.path.parent / "producer-inputs.json").write_text(json.dumps(proof))
        return R.source_gate_receipt("runtime-source", self.wanted, self.path, proof)

    def test_full_source_report_derives_actual_endpoint_count_only(self):
        receipt = self.admit()
        self.assertEqual(R.validate_receipt("runtime-source", self.wanted, receipt), [])
        self.assertEqual(R.coverage_summary(receipt), dict(expected=2, completed=2, failed=0, skipped=0))
        runtime = dict(self.wanted, kind="runtime")
        self.assertTrue(R.validate_receipt("runtime-source", runtime, receipt))

    def test_caller_one_of_one_cannot_replace_actual_coverage(self):
        receipt = self.admit()
        receipt["coverage"][0].update(expected=1, completed=1)
        self.assertTrue(any("actual producer" in error for error in R.validate_receipt("runtime-source", self.wanted, receipt)))

    def test_failed_unresolved_omitted_and_old_selected_method_reports_refuse(self):
        for case in ("status", "unresolved", "count", "duplicate", "missing rows", "old schema", "runtime proof", "tag exception"):
            with self.subTest(case=case):
                report = copy.deepcopy(self.report)
                if case == "status": report["gate"]["status"] = "failed"
                elif case == "unresolved": report["endpoints"][0]["runtime_status"] = "unresolved"
                elif case == "count": report["summary"]["endpoints"] = 1
                elif case == "duplicate": report["endpoints"][1]["endpoint"] = report["endpoints"][0]["endpoint"]
                elif case == "missing rows": report.pop("endpoints")
                elif case == "old schema": report["schema"] = 1
                elif case == "runtime proof": report["gate"]["runtime_proof_required"] = False
                else: report["endpoints"][1]["method"] = "fit"
                with self.assertRaises(ValueError): self.admit(report)

    def test_missing_modified_or_relabelled_provenance_refuses(self):
        receipt = self.admit()
        (self.path.parent / "producer-inputs.json").unlink()
        self.assertTrue(R.validate_receipt("runtime-source", self.wanted, receipt))
        for field in ("source_revision", "input_fingerprint", "report_sha256", "run_id"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit(proof_change={field: ""})
        receipt = self.admit()
        self.path.write_text("changed")
        self.assertTrue(R.validate_receipt("runtime-source", self.wanted, receipt))

    def test_current_input_closure_is_required_not_only_fingerprint_label(self):
        self.wanted["inputs"] = dict(self.inputs, changed=True)
        with self.assertRaisesRegex(ValueError, "input closure"):
            self.admit()


class CandidateReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "gate-smoke.json"
        self.identity = dict(source="candidate", vendor="nvidia", version="0.8.57", source_commit="a" * 40,
                             release_modes=["identical"], wheels={"mojolearn-core": "c" * 64, "mojolearn-nvidia": "d" * 64})
        inputs = dict(configuration={"artifacts:nvidia": dict(self.identity["wheels"]), "release_modes": ["identical"]})
        self.wanted = dict(kind="candidate", source_revision="a" * 40,
                           fingerprint=R.hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                           coverage=list(R.CANDIDATE_COVERAGE), inputs=inputs)
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
                     kind="candidate", source_revision="a" * 40, input_fingerprint=self.wanted["fingerprint"], run_id="actual-gate-run")
        proof.update(provenance_change or {})
        (self.path.parent / "producer-inputs.json").write_text(json.dumps(proof))
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

    def test_existing_receipt_reparses_raw_gate_and_rejects_caller_counts(self):
        receipt = self.admit()
        receipt["coverage"][0].update(expected=1, completed=1)
        self.assertTrue(any("actual producer" in error for error in
                            R.validate_receipt("candidate-nvidia", self.wanted, receipt)))
        receipt = self.admit()
        self.gate["measure"]["pending"] = 1
        self.path.write_text(json.dumps(self.gate))
        # Even re-hashing the edited raw report cannot retain the original
        # controller stamp or turn failed producer coverage into a pass.
        receipt["producer_report"]["sha256"] = R.sha256(self.path)
        receipt["evidence"][0]["sha256"] = R.sha256(self.path)
        self.assertTrue(R.validate_receipt("candidate-nvidia", self.wanted, receipt))

    def test_candidate_must_match_declared_artifact_input_closure(self):
        self.wanted["inputs"]["configuration"]["artifacts:nvidia"]["mojolearn-core"] = "f" * 64
        self.wanted["fingerprint"] = R.hashlib.sha256(json.dumps(self.wanted["inputs"], sort_keys=True,
                                                                separators=(",", ":")).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "artifacts/modes"):
            self.admit()

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
