"""tools/apple_github_build.py and .github/workflows/apple-release-build.yml: free standard macOS labels only, the
Darwin plan, the job's URL map, the records the Mac promotes, Metal usability, and the release / measurement-build
wiring. File-only: nothing compiles, nothing is dispatched."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import yaml

import apple_github_build as agb

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows" / agb.WORKFLOW


class Labels(unittest.TestCase):
    def test_only_free_standard_labels(self):
        self.assertEqual(agb.RUNNER_LABELS, ("macos-15", "macos-14"))
        for bad in ("macos-15-xlarge", "macos-14-large", "self-hosted", "macos-latest-xlarge"):
            with self.assertRaises(SystemExit):
                agb.partition(bad)
        self.assertEqual(agb.partition("macos-15"), "apple/gha-macos15")

    def test_workflow_is_manual_and_pins_free_labels(self):
        wf = yaml.safe_load(WF.read_text())
        on = wf.get("on", wf.get(True))
        self.assertEqual(set(on), {"workflow_dispatch"})
        self.assertEqual(on["workflow_dispatch"]["inputs"]["runner"]["default"], "macos-15")
        text = WF.read_text()
        self.assertNotIn("xlarge", text.replace("never a larger/xlarge label", ""))
        self.assertIn("macos-15|macos-14) ;;", text)
        for name, job in wf["jobs"].items():
            runs = str(job["runs-on"])
            self.assertTrue(runs in ("ubuntu-24.04", "${{ needs.plan.outputs.runner }}"), (name, runs))
        # the map URL is never interpolated into a run line
        for job in wf["jobs"].values():
            for step in job.get("steps", []):
                self.assertNotIn("inputs.map_url", str(step.get("run", "")))


class Plan(unittest.TestCase):
    def test_rows_cover_the_darwin_plan(self):
        import merged_bindings
        rows = agb.plan_rows(ROOT, ["identical"])
        want = [(d, stem) for d in ("identical", "host")
                for stem, _, _ in merged_bindings.plan(d, "darwin", True, "release", repo=ROOT)]
        self.assertEqual([(r["dir"], r["stem"]) for r in rows], want)
        self.assertEqual(len({r["key"] for r in rows}), len(rows))
        some = agb.plan_rows(ROOT, ["identical"], {f"{rows[0]['dir']}:{rows[0]['stem']}"})
        self.assertEqual(len(some), 1)

    def test_outputs_are_the_release_paths(self):
        self.assertEqual(agb.output_of("host", "_x_host"), "python/mojolearn/host/_x_host.so")
        self.assertEqual(agb.output_of("identical", "_x"), "python/mojolearn/identical/_x.so")
        self.assertEqual(agb.output_of("fast", "_x"), "python/mojolearn/_x.so")


class JobMap(unittest.TestCase):
    def test_keeps_gets_and_only_this_jobs_slots(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            body = "#partition\tapple/gha-macos15\n#leg\tL\nget\t" + "a" * 64 + "\thttps://g\n" + "".join(
                f"put\t{i:03d}\thttps://p{i}\n" for i in range(12))
            (td / "map.tsv").write_text(body)
            (td / "event.json").write_text(json.dumps({"inputs": {"map_url": (td / "map.tsv").as_uri()}}))
            agb.cmd_job_map(SimpleNamespace(event=str(td / "event.json"), index=1, dest=str(td / "out" / "urls.tsv")))
            out = (td / "out" / "urls.tsv").read_text().splitlines()
            puts = [x.split("\t")[1] for x in out if x.startswith("put")]
            self.assertEqual(puts, ["%03d" % i for i in range(agb.SLOTS_PER_JOB, 2 * agb.SLOTS_PER_JOB)])
            self.assertEqual(sum(x.startswith("get") for x in out), 1)


class Records(unittest.TestCase):
    def test_provenance_columns(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "provenance"
            p.mkdir()
            (p / "provenance.tsv").write_text("2026-11-27T00:00:00Z\tbindings/merged_build.sh\tmiss+built-cached\t"
                                              + "b" * 64 + "\t12.0\tx=y\n"
                                              "2026-11-27T00:00:01Z\tbindings/merged_build.sh\thit\t-\t0.1\t-\n")
            rows = agb.provenance_rows(td)
            self.assertEqual(rows[0], ("bindings/merged_build.sh", "miss+built-cached", "b" * 64))
            self.assertEqual(rows[1][2], "")

    def test_metal_usable_reads_the_gpu_side(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "results.json"
            p.write_text(json.dumps({"sides": {"gpu": {"ran": True, "clean": {"state": "IDENTICAL"}}}}))
            self.assertTrue(agb.metal_usable(p))
            p.write_text(json.dumps({"sides": {"gpu": {"ran": False}, "cpu": {"ran": True}}}))
            self.assertFalse(agb.metal_usable(p))
            self.assertFalse(agb.metal_usable(Path(td) / "missing.json"))

    def test_merged_key_is_the_build_scripts(self):
        import hashlib
        h = hashlib.sha256((ROOT / "bindings/merged_groups.json").read_bytes()
                           + (ROOT / "tools/merged_bindings.py").read_bytes()).hexdigest()
        self.assertEqual(agb.merged_key(ROOT), h)
        self.assertIn('MOJOLEARN_MERGED_KEY=$(cat bindings/merged_groups.json tools/merged_bindings.py | shasum -a 256',
                      (ROOT / "packaging/macos/build_release_wheel.sh").read_text())


class Wiring(unittest.TestCase):
    def test_groups_only_mode_is_not_keyed(self):
        import bincache
        self.assertTrue("MOJOLEARN_RELEASE_MACOS_GROUPS".startswith(bincache.NON_BUILD_PREFIXES))
        self.assertIn('GROUPS_ONLY="${MOJOLEARN_RELEASE_MACOS_GROUPS:-}"',
                      (ROOT / "packaging/macos/build_release_wheel.sh").read_text())

    def test_release_default_is_github(self):
        import release
        args = release.release_parser().parse_args(["0.9.1"])
        self.assertEqual(args.macos_build_backend, "github")
        args = release.release_parser().parse_args(["0.9.1", "--macos-build-backend", "laptop"])
        self.assertEqual(args.macos_build_backend, "laptop")

    def test_measurement_build_adds_the_apple_step_to_an_a_set_only(self):
        import release
        import release_candidate as rc
        a = SimpleNamespace(pack_here=False, no_apple=False)
        steps = rc.run_steps(a, "", release)
        self.assertEqual([s for s in steps if s in rc.STEPS], list(rc.STEPS))
        self.assertIn("macos-build", steps)
        self.assertNotIn("macos-build", rc.run_steps(a, "-D X=1", release))
        self.assertNotIn("macos-build", rc.run_steps(SimpleNamespace(pack_here=False, no_apple=True), "", release))
        self.assertNotIn("macos-build", rc.run_steps(SimpleNamespace(pack_here=True, no_apple=False), "", release))
        needs = {s: n for s, _, n, _ in release.STEP_TABLE}
        skipped = {"readiness"}
        for need in needs["macos-build"]:
            self.assertTrue(need in rc.STEPS or need in skipped, need)


if __name__ == "__main__":
    unittest.main()
