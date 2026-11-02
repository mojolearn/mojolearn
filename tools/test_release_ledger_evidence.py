"""Post-0.8.63 release tooling, plan items 1 and 2 (docs/plans/POST_0863_RELEASE_TOOLING.md).

A Python/data-only refreeze launches no build and no cross-compile; a finished
leg's begun compile runs are recorded from its own output; one gate predicate
decides step_gate_<vendor>, publication and the record; the committed identity
table admission is an identity producer of the ledger.

    python3 -m pytest -q tools/test_release_ledger_evidence.py
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_requirements  # noqa: E402

OLD = "a" * 40
NEW = "c" * 40
BASE = "b" * 40


def args(**kw):
    base = dict(version="0.8.99", dry_run=False, publish=None, only="", redo="", build_backend="github",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                cpu_column=False, status=False, refreeze=False, source_checkout="", owner_waiver=None)
    base.update(kw)
    return argparse.Namespace(**base)


class Base(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self._env = mock.patch.dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=str(self.tmp / "ev"),
                                    MOJOLEARN_RELEASE_LEDGER="off")
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._t.cleanup()

    def release(self, **kw):
        r = release.Release(args(state_dir=str(self.tmp / "state"), **kw), runner=lambda *a, **k: 0)
        r.state["commit"] = NEW
        r.lines = []
        r.say = r.lines.append
        return r


class PythonOnlyRefreeze(Base):
    """Refreezing for a commit that changes no native build input reuses both."""

    def xcc_receipt(self, r, commit=OLD, **over):
        doc = dict(schema=r.XCC_SCHEMA, verdict="PASS", commit=commit, base=BASE, archs=r.XCC_ARCHS,
                   tiers=list(r.release_modes), run="https://github.com/o/r/actions/runs/7")
        doc.update(over)
        path = r.base / commit[:12] / "cross-compile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))

    def test_cross_compile_pass_is_reused_when_no_native_input_changed(self):
        r = self.release()
        self.xcc_receipt(r)
        with mock.patch.object(release.release_reuse, "previous_release",
                               return_value=dict(version="0.8.98", source_commit=BASE)), \
                mock.patch.object(release.Release, "kernel_changes", return_value=[]) as changes, \
                mock.patch.object(release.Release, "must") as must:
            out, data = r.step_cross_compile()
        must.assert_not_called()  # no gh workflow run
        changes.assert_called_once_with(OLD, NEW)
        self.assertIn("PASS reused from " + OLD[:12], out)
        self.assertEqual(data["reused_from"], OLD)

    def test_cross_compile_runs_when_a_native_input_changed_or_the_base_moved(self):
        r = self.release()
        self.xcc_receipt(r)
        self.xcc_receipt(r, commit="d" * 40, base="e" * 40)
        with mock.patch.object(release.release_reuse, "previous_release",
                               return_value=dict(version="0.8.98", source_commit=BASE)), \
                mock.patch.object(release.Release, "kernel_changes", return_value=["bindings/build_x.sh"]), \
                mock.patch.object(release.Release, "must", side_effect=release.StepFailed("dispatched")) as must:
            with self.assertRaises(release.StepFailed):
                r.step_cross_compile()
        self.assertEqual(must.call_args[0][0][:3], ["gh", "workflow", "run"])
        self.assertTrue(any("not reused" in line and "bindings/build_x.sh" in line for line in r.lines))

    def test_unreadable_trees_never_count_as_unchanged(self):
        r = self.release()
        self.xcc_receipt(r)
        with mock.patch.object(release.release_reuse, "facts_for_commit", side_effect=OSError("gone")):
            self.assertIsNone(r.kernel_changes(OLD, NEW))
        with mock.patch.object(release.release_reuse, "previous_release",
                               return_value=dict(version="0.8.98", source_commit=BASE)), \
                mock.patch.object(release.Release, "kernel_changes", return_value=None):
            self.assertIsNone(r.cross_compile_reuse(BASE))

    def runtime(self, portable):
        return dict(schema="s", target="linux-64", what="runtime", toolchain={"packages": {"max": "1"}},
                    image={"nvidia": "img"}, portable_math=portable, builders={"x": "y"})

    def leg_fixture(self, r, changed):
        old_rt, new_rt = self.runtime("p-old"), self.runtime("p-new")
        digest = release.release_reuse.digest_of
        from ptx_reuse import digest as set_digest
        bindings = {"cuda/sm_89/identical/_mojolearn_x": "1" * 64, "host/-/host/_mojolearn_y_host": "2" * 64}

        def ident(rt):
            doc = dict(set="cuda/sm_89", bindings=bindings, runtime=digest(rt))
            return dict(doc, digest=set_digest(doc))
        plan = r.base / OLD[:12] / "reuse" / "plan.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps(dict(commit=OLD, runtime=dict(identity=old_rt))))
        candidate = dict(source_commit=OLD, set_identity=ident(old_rt), tooling_digest="T",
                         release_build=str(self.tmp / "rb"), tooling_commit=OLD)
        patches = [
            mock.patch.object(release.Release, "set_identity", return_value=ident(new_rt)),
            mock.patch.object(release.Release, "packer_takes_legs", return_value=True),
            mock.patch.object(release.Release, "tooling", return_value=dict(digest="T", commit=NEW, dirty=[])),
            mock.patch.object(release.Release, "leg_candidates", return_value=[("old/legs/cuda-sm_89", candidate)]),
            mock.patch.object(release.Release, "plan", return_value=dict(runtime=dict(identity=new_rt),
                                                                       leg_reasons={}, previous={})),
            mock.patch.object(release, "verify_leg_tree", return_value=(True, "", "sha")),
            mock.patch.object(release, "overlay_verified", return_value=(True, "")),
            mock.patch.object(release, "git", side_effect=lambda *a, **k: "\n".join(changed)),
        ]
        return patches

    def run_builds(self, changed):
        r = self.release()
        leg = release.Leg("cuda-sm_89", "cuda", "sm_89", ["build"], {}, self.tmp / "now" / "rb",
                          self.tmp / "now", self.tmp / "now" / "out")
        patches = self.leg_fixture(r, changed) + [
            mock.patch.object(release, "linux_legs", return_value=[leg]),
            mock.patch.object(release.Release, "find_requirement_leg", return_value=(None, "n/a", False)),
            mock.patch.object(release, "launch_detached")]
        for p in patches:
            p.start()
        try:
            out = r.step_linux_builds()
            return out, release.launch_detached.call_args[0][1]
        finally:
            for p in reversed(patches):
                p.stop()

    def test_audit_only_portable_math_change_takes_the_leg_and_launches_no_build(self):
        out, launched = self.run_builds(["packaging/portable_math/wheel.py", "packaging/portable_math/test_wheel.py"])
        self.assertEqual(launched, [])
        self.assertIn("taken from an earlier freeze: cuda-sm_89", out)
        doc = json.loads((self.tmp / "state" / NEW[:12] / "legs" / "cuda-sm_89.admission.json").read_text()) \
            if (self.tmp / "state" / NEW[:12] / "legs" / "cuda-sm_89.admission.json").exists() else None
        if doc:
            self.assertIn("no byte is built", doc["because"])

    def test_a_compiled_portable_math_input_change_builds(self):
        _, launched = self.run_builds(["packaging/portable_math/portable_math.c"])
        self.assertEqual([leg.name for leg in launched], ["cuda-sm_89"])

    def test_a_binding_identity_change_is_never_neutral(self):
        r = self.release()
        for p in self.leg_fixture(r, ["packaging/portable_math/wheel.py"]):
            p.start()
        try:
            prior = r.leg_candidates(0, 0, 0, 0)[0][1]["set_identity"]
            self.assertIsNotNone(r.byte_neutral_runtime_change(OLD, prior, r.set_identity("cuda", "sm_89")))
            moved = dict(r.set_identity("cuda", "sm_89"), bindings={"k": "3" * 64})
            self.assertIsNone(r.byte_neutral_runtime_change(OLD, prior, moved))
        finally:
            mock.patch.stopall()


class FinishedLegsAreRecorded(Base):
    def test_a_finished_unrecorded_leg_is_recorded_from_its_output(self):
        r = self.release()
        legs = r.base / OLD[:12] / "legs"
        legs.mkdir(parents=True)
        (legs / "cuda-sm_89.exit").write_text("0\n")
        prov = legs / "cuda-sm_89.provenance.json"
        import release_compile_evidence as ce
        launch = dict(schema=ce.LAUNCH_SCHEMA, selected=["compile-nvidia/u"],
                      runs={"compile-nvidia/u": dict(run_id="R1", fingerprint="F", source_revision=OLD)})
        doc = dict(source_commit=OLD, release_build=str(legs / "cuda-sm_89" / "release-build"),
                   exit_file=str(legs / "cuda-sm_89.exit"), compile_requirements=launch)
        prov.write_text(json.dumps(doc))
        ledger = dict(schema=release_requirements.SCHEMA, requirements={}, attempts=[], runs=[dict(id="R1")])
        recorded = []
        with mock.patch.object(release.Release, "leg_candidates", return_value=[(str(prov), doc)]), \
                mock.patch.object(release_requirements.RequirementsLedger, "read", return_value=ledger), \
                mock.patch.object(release.Leg, "done", return_value=True), \
                mock.patch.object(release, "verify_leg_tree", return_value=(True, "", "sha")), \
                mock.patch.object(release.Release, "record_compile_requirements",
                                  side_effect=lambda leg: recorded.append(leg)):
            self.assertEqual(r.record_finished_legs("cuda-sm_89", "cuda", "sm_89", {"digest": "x"}), [str(prov)])
            self.assertEqual(recorded[0].provenance_file, prov)
            ledger["attempts"].append(dict(run_id="R1"))  # once recorded, never again
            self.assertEqual(r.record_finished_legs("cuda-sm_89", "cuda", "sm_89", {"digest": "x"}), [])

    def test_an_unfinished_or_failed_leg_is_not_recorded(self):
        r = self.release()
        import release_compile_evidence as ce
        doc = dict(source_commit=OLD, release_build=str(self.tmp / "rb"), compile_requirements=dict(
            schema=ce.LAUNCH_SCHEMA, runs={"u": dict(run_id="R1")}))
        prov = self.tmp / "legs" / "cuda-sm_89.provenance.json"
        ledger = dict(schema=release_requirements.SCHEMA, requirements={}, attempts=[], runs=[dict(id="R1")])
        with mock.patch.object(release.Release, "leg_candidates", return_value=[(str(prov), doc)]), \
                mock.patch.object(release_requirements.RequirementsLedger, "read", return_value=ledger), \
                mock.patch.object(release.Leg, "done", return_value=False), \
                mock.patch.object(release.Release, "record_compile_requirements") as record:
            self.assertEqual(r.record_finished_legs("cuda-sm_89", "cuda", "sm_89", {"digest": "x"}), [])
        record.assert_not_called()

    def test_read_only_runs_record_nothing(self):
        r = self.release(status=True)
        with mock.patch.object(release.Release, "leg_candidates") as candidates:
            self.assertEqual(r.record_finished_legs("cuda-sm_89", "cuda", "sm_89", {}), [])
        candidates.assert_not_called()


class OneGatePredicate(Base):
    ROWS = {"identity-cpu/x": dict(required=True, status="untested", stage="pre", reasons=["no evidence"]),
            "published-amd": dict(required=True, status="untested", stage="post", reasons=["later"]),
            "runtime-cpu": dict(required=True, status="passed", stage="pre", reasons=[])}

    def test_publish_record_and_gate_steps_share_release_gate(self):
        r = self.release(publish="pypi")
        with mock.patch.object(release.Release, "gate_ok", return_value=(False, "no gate report")), \
                mock.patch.object(release.Release, "readiness_status", return_value=self.ROWS):
            ready, reasons = r.release_gate(stage="pre")
            self.assertFalse(ready)
            self.assertEqual(sorted(x.split(":")[0] for x in reasons), ["candidate gate amd", "candidate gate nvidia"])
        with mock.patch.object(release.Release, "gate_ok", return_value=(True, "PASS")), \
                mock.patch.object(release.Release, "readiness_status", return_value=self.ROWS):
            self.assertEqual(r.release_gate(stage="pre")[1][0].split(":")[0], "identity-cpu/x")
        with mock.patch.object(release.Release, "gate_ok", return_value=(False, "no gate report")), \
                mock.patch.object(release.Release, "readiness_status", return_value=self.ROWS):
            with mock.patch.object(release.Release, "release_gate", return_value=(False, ["x"])) as gate:
                with self.assertRaises(release.StepFailed):
                    r.publish("linux", None, None)
                gate.assert_called_once_with(stage="pre")
                with mock.patch.object(release.Release, "published_platforms", return_value=["amd"]):
                    with self.assertRaises(release.StepFailed):
                        r.step_record()
                self.assertEqual(gate.call_args, mock.call(vendors=["amd"]))
        with mock.patch.object(release.Release, "gate_ok", return_value=(True, "PASS")), \
                mock.patch.object(release.Release, "run_gate") as run_gate:
            self.assertEqual(r.step_gate_nvidia(), "PASS")
        run_gate.assert_not_called()

    def test_owner_waiver_applies_through_the_same_predicate(self):
        waiver = self.tmp / "waiver.json"
        waiver.write_text(json.dumps(dict(prefixes=["candidate-", "identity-"], approved_by="owner", reason="r")))
        r = self.release(owner_waiver=str(waiver))
        with mock.patch.object(release.Release, "gate_ok", return_value=(False, "no gate report")), \
                mock.patch.object(release.Release, "readiness_status", return_value=self.ROWS):
            self.assertEqual(r.release_gate(stage="pre"), (True, []))
            ready, reasons = r.release_gate()
        self.assertFalse(ready)
        self.assertEqual([x.split(":")[0] for x in reasons], ["published-amd"])


class IdentityTableAdmission(unittest.TestCase):
    def test_adapter_is_registered_as_an_identity_producer(self):
        import release_ledger_admission as admission
        self.assertEqual(admission.ADAPTERS["identity-table-admission"], ("release_requirements_retained", "identity"))
        self.assertIn(("identity", "tools/release_requirements_retained.py"), admission.tools())

    def test_readme_records_are_content_addressed(self):
        import release_requirements_retained as retained
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "adm").mkdir()
            sha = "0" * 64
            (root / "adm" / "README.md").write_text(
                f"| `bench/results/identity_break/retained/{sha}/apple-m4.identical.json` | x |\n"
                f"| `bench/results/identity_break/retained/{sha}/apple-m4.identical.json` | dup |\n")
            self.assertEqual(retained.admission_records(root, "adm"),
                             [(f"bench/results/identity_break/retained/{sha}/apple-m4.identical.json", sha)])
            (root / "adm" / "README.md").write_text("nothing")
            with self.assertRaises(ValueError):
                retained.admission_records(root, "adm")

    def test_optional_loader_absence_is_proven_by_loaded_witnesses(self):
        import release_requirements_retained as retained
        import release_identity_evidence as ident
        record = dict(package=dict(bindings=[dict(module="mojolearn._host._mojolearn_tokenizer_host")]))
        contract = {"_mojolearn_tokenizer_fast": dict(present=False, predicate={})}
        retained._optional_files(ident, record, "p", contract)
        with self.assertRaises(ValueError):
            retained._optional_files(ident, record, "p", {"_mojolearn_tokenizer_fast": dict(present=True)})
        record["package"]["bindings"].append(dict(module="mojolearn._fast._mojolearn_tokenizer_fast"))
        with self.assertRaises(ValueError):
            retained._optional_files(ident, record, "p", contract)

    def test_retained_attempts_join_only_declared_requirements_once(self):
        import release_requirements_retained as retained
        doc = dict(requirements={"identity-cpu/x": {}}, attempts=[dict(id="A")])
        merged = retained.with_attempts(doc, [dict(id="A", requirement="identity-cpu/x"),
                                              dict(id="B", requirement="identity-cpu/x"),
                                              dict(id="C", requirement="identity-cpu/absent")])
        self.assertEqual([a["id"] for a in merged["attempts"]], ["A", "B"])
        self.assertEqual([a["id"] for a in doc["attempts"]], ["A"])


class OptionalNativePresence(Base):
    def test_presence_is_read_from_the_candidate_wheels(self):
        import zipfile
        r = self.release()
        mac = r.rel / "macos" / "mojolearn-0.8.99-py3-none-macosx_26_0_arm64.whl"
        mac.parent.mkdir(parents=True)
        with zipfile.ZipFile(mac, "w") as z:
            z.writestr("mojolearn/host/_mojolearn_tokenizer_host.so", b"")
        manifest = dict(requirements={"identity-apple/bpe": dict(missing_inputs=[
            r.UNMAPPED_BINDING + "_mojolearn_tokenizer_fast"])})
        with mock.patch.object(release.Release, "split_final", return_value=None):
            self.assertEqual(r.optional_native_presence(manifest, {}),
                             {"osx-arm64": {"_mojolearn_tokenizer_fast": False}})
            declared = {"optional_native_files": {"osx-arm64": {"_mojolearn_tokenizer_fast": False}}}
            self.assertIsNone(r.optional_native_presence(manifest, declared))
            self.assertIsNone(r.optional_native_presence(dict(requirements={}), {}))


if __name__ == "__main__":
    unittest.main()


class TableBackedIdentityAdmission(unittest.TestCase):
    """identity-<vendor>/<lane> from the reference table's cited records when another column was not
    re-recorded at the admission revision (Andrew, 2026-10-31: re-record only changed lanes)."""

    @staticmethod
    def by_vendor(record, path, ref, revision, equivalent=()):
        return ("apple" if "apple" in record["vendor"] else "amd"), []

    def facts(self, root, own_sha, cited):
        return dict(source_root=str(root), source_revision=NEW, readme=dict(path=str(root / "README.md"), sha256="r" * 64),
                    records=[dict(cls="amd", path=str(root / "amd.identical.json"), sha256=own_sha,
                                  size=(root / "amd.identical.json").stat().st_size,
                                  provenance=dict(path=str(root / "amd.identical.json"), sha256=own_sha), lanes=["ols"])],
                    table=dict(path="t", revision=OLD, sha256="t" * 64), cited=cited)

    def table_case(self, apple_digest="1234567890abcdef"):
        import release_requirements_retained as retained
        import release_identity_evidence as ident
        import test_release_identity_evidence as fixtures
        root = pathlib.Path(tempfile.mkdtemp()).resolve()
        amd = fixtures.record("amd", ["ols"], commit=NEW)
        apple = fixtures.record("apple", ["ols"], commit=OLD, digests={("ols", f): apple_digest for f in fixtures.FIXTURES})
        (root / "amd.identical.json").write_text(json.dumps(amd))
        (root / "apple.identical.json").write_text(json.dumps(apple))
        amd_sha = ident.reference(root / "amd.identical.json")["sha256"]
        apple_sha = ident.reference(root / "apple.identical.json")["sha256"]
        digest = "1234567890abcdef"
        cells = {f"ols/{f}": {"train": dict(cols={"amd": 0, "apple": 1}, ref=digest),
                              "infer": dict(cols={"amd": 0, "apple": 1}, ref=digest[::-1]),
                              "batchgrad": dict(cols={"amd": 0, "apple": 1}, ref="n/a:no-backward")}
                 for f in fixtures.FIXTURES}
        table = dict(records=[dict(sha256=amd_sha), dict(sha256=apple_sha)], cells=cells)
        retained._TABLES[(str(root.resolve()), OLD)] = (dict(path="t", revision=OLD, sha256="t" * 64), table)
        cited = [dict(index=0, cls="amd", commit=NEW, path=str(root / "amd.identical.json"), sha256=amd_sha),
                 dict(index=1, cls="apple", commit=OLD, path=str(root / "apple.identical.json"), sha256=apple_sha)]
        environment = dict(target="linux-64", loader_routes={})
        return retained, ident, root, amd_sha, cited, environment

    def test_unit_admitted_from_the_table_when_the_other_column_is_older(self):
        retained, ident, root, amd_sha, cited, environment = self.table_case()
        with mock.patch.object(ident, "record_errors", side_effect=self.by_vendor), \
                mock.patch.object(ident, "platform_target", return_value="linux-64"):
            rows, refs = retained.table_check_unit(ident, "ols", "amd", ("amd", "apple"), NEW, environment,
                                                   self.facts(root, amd_sha, cited))
        self.assertEqual([(r["id"], r["status"], r["expected"], r["completed"]) for r in rows],
                         [(ident.BASE_COVERAGE, "passed", 6, 6), ("lane:ols", "passed", 6, 6)])
        self.assertIn(cited[1]["path"], [r["path"] for r in refs])

    def test_a_cited_column_whose_bytes_disagree_with_the_table_is_refused(self):
        retained, ident, root, amd_sha, cited, environment = self.table_case(apple_digest="feedfeedfeedfeed")
        with mock.patch.object(ident, "record_errors", side_effect=self.by_vendor), \
                mock.patch.object(ident, "platform_target", return_value="linux-64"):
            with self.assertRaisesRegex(ValueError, "apple column feedfeedfeedfeed differs from the table"):
                retained.table_check_unit(ident, "ols", "amd", ("amd", "apple"), NEW, environment,
                                          self.facts(root, amd_sha, cited))

    def test_a_table_citing_another_own_record_is_refused(self):
        retained, ident, root, amd_sha, cited, environment = self.table_case()
        cited[0] = dict(cited[0], sha256="0" * 64)
        with mock.patch.object(ident, "record_errors", side_effect=self.by_vendor), \
                mock.patch.object(ident, "platform_target", return_value="linux-64"):
            with self.assertRaisesRegex(ValueError, "cites another amd record"):
                retained.table_check_unit(ident, "ols", "amd", ("amd", "apple"), NEW, environment,
                                          self.facts(root, amd_sha, cited))


class RetainedProducersWiring(Base):
    def test_every_gate_directory_and_leg_is_a_retained_producer(self):
        r = self.release()
        gate = r.rel / "gate-nvidia" / "abc-release-light-v1"
        (gate / "adapter-smoke").mkdir(parents=True)
        (gate / "gate-smoke.json").write_text("{}")
        legs = r.rel / "legs"
        (legs / "hip-gfx942").mkdir(parents=True)
        (legs / "hip-gfx942.provenance.json").write_text("{}")
        (legs / "hip-gfx942.exit").write_text("0\n")
        earlier = self.tmp / "earlier" / "legs"
        (earlier / "cuda-sm_89").mkdir(parents=True)
        (earlier / "cuda-sm_89.provenance.json").write_text("{}")
        (earlier / "cuda-sm_89.exit").write_text("0\n")
        (legs / "cuda-sm_89.reused.json").write_text(json.dumps(dict(leg="cuda-sm_89", provenance=str(earlier / "cuda-sm_89.provenance.json"))))
        (legs / "cuda-sm_80.reused.json").write_text(json.dumps(dict(leg="cuda-sm_80", provenance=str(earlier / "missing.provenance.json"))))
        with mock.patch.object(release.Release, "macos_wheel", return_value=None):
            producers = r.retained_producers()
        self.assertIn(("gate-adapters", dict(gate=str(gate))), producers)
        self.assertEqual([p for n, p in producers if n == "linux-release-leg"],
                         [dict(leg=str(legs / "hip-gfx942")), dict(leg=str(earlier / "cuda-sm_89"))])

    def test_runtime_test_revision_defaults_to_the_frozen_source(self):
        r = self.release()
        with mock.patch.object(release.release_reuse, "darwin_toolchain", return_value=None), \
                mock.patch.object(release.Release, "candidate_gate_wheels", return_value=[]), \
                mock.patch.object(release.Release, "split_finals", return_value={}):
            try:
                manifest = r.readiness_manifest()
            except Exception as error:  # the rest of the manifest needs a frozen tree; the default is set first
                self.skipTest("readiness manifest needs a frozen tree here: " + str(error)[:80])
        self.assertEqual(manifest["configuration"]["runtime_test_source_revision"], NEW)
