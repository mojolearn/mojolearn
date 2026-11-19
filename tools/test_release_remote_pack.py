# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""A measurement build's pack on GitHub (lane fx-pack-offload, 2026-11-19): tools/release_remote_pack.py (the bundle,
its safe extraction, the runner's command and artifact, the Mac's install), the records-only collector receipt
(tools/release_github_build.py), the release.py pack modes (candidate_pack, RemoteLeg) and the candidate's --pack
options and candidate.json `pack`, and the two workflows. No network, no git remote, no gh, no build."""
import datetime as dt
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import release  # noqa: E402
import release_candidate as RC  # noqa: E402
import release_github_build as G  # noqa: E402
import release_remote_pack as RP  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
BASE_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def write(path, text="x"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class FakeRelease:
    """The parts of release.Release the bundle reads."""

    def __init__(self, work, state, base=None, plan=None, evidence=None):
        self.base, self.state, self._base, self._plan = Path(work), state, base, plan or {}
        self.commit, self.version = state["commit"], state["version"]
        self.evidence = Path(evidence or work)

    @property
    def rel(self):
        return self.base / self.commit[:12]

    def candidate_base(self):
        return self._base

    def plan(self):
        return self._plan


FakeRelease.__module__ = "release"   # release_remote_pack.release_module: the running Release's module


def a_state(**candidate):
    cand = dict(version="0.8.67.dev0+0123456789ab", commit=SHA, defines="", label="", targets=["sm_89", "sm_80"],
                base_version="0.8.67", owed_arm=None, tooling_sha=SHA, tooling_ref="lane/x", pack="github")
    cand.update(candidate)
    return dict(version=cand["version"], commit=SHA, release_modes=["identical"], candidate=cand,
                steps={s: dict(done=True) for s in ("reuse-plan", "linux-builds", "linux-wait", "linux-assemble",
                                                       "linux-pack")},
                failures={"linux-pack": dict(error="audit.sh exited 125")}, remote_pack=dict(run_id=1))


class Records(unittest.TestCase):
    def test_keep_suffixes_are_the_candidate_prune_rule(self):
        self.assertEqual(RP.KEEP_SUFFIXES, RC.KEEP_SUFFIXES)

    def test_records_filters_payload_attempts_source_and_skips(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td)
            keep = [write(w / "state.json"), write(w / "c" / "legs" / "cuda-sm_89.exit", "0"),
                    write(w / "c" / "legs" / "cuda-sm_89" / "GHA" / "jobs.tsv")]
            write(w / "c" / "legs" / "cuda-sm_89" / "x.so")
            write(w / "c" / "legs" / "cuda-sm_89.failed-20261119" / "a.json")
            write(w / "c" / "legs" / "cuda-sm_89.log.failed-20261119")
            write(w / "source" / "c" / "README.md")
            write(w / "c" / "linux" / "final" / "candidate.json")
            got = RP.records(w, skip=(w / "c" / "linux" / "final",))
            self.assertEqual(sorted(got), sorted(keep))

    def test_runner_state_unmarks_the_tree_steps(self):
        doc = RP.runner_state(a_state())
        self.assertEqual(sorted(doc["steps"]), ["linux-builds", "reuse-plan"])
        self.assertNotIn("linux-pack", doc["failures"])
        self.assertNotIn("remote_pack", doc)
        self.assertEqual(a_state()["steps"].keys(), a_state()["steps"].keys())   # the Mac's state is not touched


class Bundle(unittest.TestCase):
    def make(self, td, base=None, record_dir=None):
        work = Path(td) / "Users" / "me" / "rc" / "0.8.67.dev0+0123456789ab"
        rel = work / SHA[:12]
        write(work / "state.json", "{}")
        write(work / "ledger" / "pass" / "a.json", "{}")
        write(rel / "reuse" / "plan.json", "{}")
        write(rel / "reuse" / "sets" / "cuda" / "sm_89" / "lib.so", "payload")
        write(rel / "legs" / "cuda-sm_89.exit", "0")
        write(rel / "legs" / "cuda-sm_89" / "GHA" / "records-only.json", "{}")
        write(work / "source" / SHA[:12] / "big.json", "{}")
        plan = dict(previous=dict(record_dir=str(record_dir)) if record_dir else {})
        r = FakeRelease(work, a_state(), base=base, plan=plan, evidence=Path(td) / "Users" / "me" / "ev")
        return r, work, rel

    def test_spec_and_members(self):
        with tempfile.TemporaryDirectory() as td:
            outside = Path(td) / "Users" / "me" / "ev" / "release" / "0.8.66" / "rec"
            write(outside / "release-provenance.json", "{}")
            write(outside / "huge.log", "not shipped")
            r, work, rel = self.make(td, record_dir=outside)
            spec = RP.make_spec(r, {"cuda-sm_89": "123"}, "cpk-t", None)
            self.assertEqual(spec["schema"], RP.SPEC_SCHEMA)
            self.assertEqual(spec["legs"], {"cuda-sm_89": "123"})
            self.assertIn(str(work.parent), spec["mirror"])
            self.assertIn(str(outside), spec["mirror"])
            self.assertEqual(spec["root"], str(RP.ROOT))
            arc = RP.bundle(r, spec, Path(td) / "out" / "b.tar.gz")
            with tarfile.open(arc) as tf:
                names = tf.getnames()
                self.assertIn(RP.SPEC_ARC, names)
                st = json.loads(tf.extractfile(RP.ABS + str(work / "state.json").lstrip("/")).read())
                self.assertNotIn("linux-wait", st["steps"])     # the runner's state, not the file on disk ("{}")
                targets = {str(t) for _, t in RP.safe_members(tf, spec["mirror"])}
            self.assertIn(str(rel / "legs" / "cuda-sm_89.exit"), targets)
            self.assertIn(str(rel / "reuse" / "plan.json"), targets)
            self.assertIn(str(work / "ledger" / "pass" / "a.json"), targets)
            self.assertIn(str(outside / "release-provenance.json"), targets)
            self.assertNotIn(str(outside / "huge.log"), targets)               # record dirs: .json only
            self.assertFalse(any(t.endswith(".so") for t in targets))          # no payload
            self.assertFalse(any("/source/" in t for t in targets))            # no source worktree

    def test_record_dir_inside_the_tooling_checkout_is_an_alias_not_a_member(self):
        with tempfile.TemporaryDirectory() as td:
            r, _, _ = self.make(td, record_dir=RP.ROOT / "bench" / "results" / "release_evidence")
            spec = RP.make_spec(r, {}, "cpk-t", None)
            self.assertEqual(spec["record_dirs"], [])
            self.assertNotIn(str(RP.ROOT / "bench" / "results" / "release_evidence"), spec["mirror"])

    def test_base_records_and_wheels(self):
        with tempfile.TemporaryDirectory() as td:
            bwork = Path(td) / "Users" / "me" / "rc" / "0.8.67.dev0+fedcba987654"
            bd = bwork / BASE_SHA[:12]
            write(bwork / "state.json", "{}")
            write(bd / "reuse" / "plan.json", "{}")
            write(bd / "linux" / "final" / "candidate.json", "{}")
            write(bd / "linux" / "final" / "mojolearn_core-x.whl", "wheel")
            write(bd / "legs" / "cuda-sm_80.provenance.json", "{}")
            write(bd / "legs" / "cuda-sm_80.log", "not shipped")
            base = dict(build_dir=str(bd), version="v", linux=dict(wheel="mojolearn_core-x.whl", sha256="0" * 64))
            r, _, _ = self.make(td, base=base)
            for run, wheel_in in ((None, True), ("77", False)):
                spec = RP.make_spec(r, {}, "cpk-t", run)
                arc = RP.bundle(r, spec, Path(td) / f"b{run}.tar.gz")
                with tarfile.open(arc) as tf:
                    targets = {str(t) for _, t in RP.safe_members(tf, spec["mirror"])}
                self.assertIn(str(bwork / "state.json"), targets)
                self.assertIn(str(bd / "legs" / "cuda-sm_80.provenance.json"), targets)
                self.assertNotIn(str(bd / "legs" / "cuda-sm_80.log"), targets)
                self.assertEqual(str(bd / "linux" / "final" / "mojolearn_core-x.whl") in targets, wheel_in)

    def test_safe_members_refuses_escapes(self):
        with tempfile.TemporaryDirectory() as td:
            arc = Path(td) / "bad.tar.gz"
            src = write(Path(td) / "f", "x")
            with tarfile.open(arc, "w:gz") as tf:
                tf.add(str(src), arcname=RP.ABS + "etc/passwd")
            with tarfile.open(arc) as tf, self.assertRaises(SystemExit):
                RP.safe_members(tf, [str(Path(td) / "allowed")])
            with tarfile.open(arc, "w:gz") as tf:
                tf.add(str(src), arcname="elsewhere/f")
            with tarfile.open(arc) as tf, self.assertRaises(SystemExit):
                RP.safe_members(tf, ["/"])


class BaseWheels(unittest.TestCase):
    def test_source(self):
        with tempfile.TemporaryDirectory() as td:
            bd = Path(td) / "b"
            self.assertEqual(RP.base_wheel_source(None)[0], None)
            write(bd / "linux" / "final" / "candidate.json", json.dumps(dict(pack=dict(where="local"))))
            self.assertIsNone(RP.base_wheel_source(dict(build_dir=str(bd)))[0])
            when = dt.datetime(2026, 11, 19, tzinfo=dt.timezone.utc)
            write(bd / "linux" / "final" / "candidate.json",
                  json.dumps(dict(pack=dict(where="github", run_id=4242, packed_at=when.isoformat()))))
            self.assertEqual(RP.base_wheel_source(dict(build_dir=str(bd)), today=when + dt.timedelta(days=3))[0],
                             "4242")
            old, why = RP.base_wheel_source(dict(build_dir=str(bd)), today=when + dt.timedelta(days=13))
            self.assertIsNone(old)
            self.assertIn("13 days", why)


class Runner(unittest.TestCase):
    def spec(self, **cand):
        c = dict(defines="", label="", targets=["sm_89", "sm_80"], timed_on=[], base_version="0.8.67")
        c.update(cand)
        return dict(commit=SHA, work="/Users/me/rc/0.8.67.dev0+0123456789ab", candidate=c,
                    all_targets=["sm_89", "sm_80", "gfx942"])

    def test_command_is_the_candidate_with_pack_here(self):
        cmd = RP.candidate_command(self.spec())
        self.assertEqual(cmd[1:4], [str(RP.ROOT / "tools" / "release.py"), "candidate", "--pack-here"])
        for flag in ("--no-prebuild-check", "--no-verify-smoke", "--keep-intermediates"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--work-root") + 1], "/Users/me/rc")
        self.assertEqual(cmd[cmd.index("--targets") + 1], "sm_89,sm_80")
        self.assertNotIn("--defines", cmd)
        every = RP.candidate_command(self.spec(targets=["sm_89", "sm_80", "gfx942"]))
        self.assertNotIn("--targets", every)
        arm = RP.candidate_command(self.spec(defines="A=1", label="small.mlp", timed_on=["sm_89"]))
        self.assertEqual(arm[arm.index("--defines") + 1], "A=1")
        self.assertEqual(arm[arm.index("--label") + 1], "small.mlp")
        self.assertEqual(arm[arm.index("--timed-on") + 1], "sm_89")
        # the runner's command parses as a candidate command
        a = RC.parse(arm[3:])
        self.assertTrue(a.pack_here and a.no_verify_smoke and a.keep_intermediates)

    def test_collect_and_install(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "runner" / "w"
            rel = work / SHA[:12]
            write(work / "state.json", json.dumps(dict(steps={"linux-pack": {}})))
            write(rel / "linux" / "final" / "mojolearn_core-1.whl", "wheel")
            write(rel / "linux" / "final" / "split-audit.json", json.dumps(dict(problems=[])))
            write(rel / "linux-pack.log", "packed")
            write(rel / "legs" / "cuda-sm_89" / "cuda-sm_89" / "release-build" / "build" / "build-provenance.json")
            write(rel / "legs" / "cuda-sm_89" / "cuda-sm_89" / "release-build" / "build" / "x.so")
            write(rel / "remote-pack" / "t" / "bundle.json")
            write(work / "source" / SHA[:12] / "a.json")
            out = Path(td) / "out"
            RP.collect(dict(work=str(work), commit=SHA), out)
            self.assertTrue((out / "final" / "mojolearn_core-1.whl").is_file())
            with tarfile.open(out / "records.tar.gz") as tf:
                names = set(tf.getnames())
            self.assertIn("state.json", names)
            self.assertIn(f"{SHA[:12]}/linux-pack.log", names)
            self.assertIn(f"{SHA[:12]}/legs/cuda-sm_89/cuda-sm_89/release-build/build/build-provenance.json", names)
            self.assertFalse(any(n.endswith(".so") or "/source/" in n or n.startswith("source/")
                                 or "/linux/final/" in n or "/remote-pack/" in n for n in names))
            # the Mac installs it
            mac = Path(td) / "mac" / "w"
            write(mac / SHA[:12] / "linux" / "old.whl")
            r = FakeRelease(mac, a_state())
            RP.install(r, out, "t")
            self.assertTrue((mac / SHA[:12] / "linux" / "final" / "mojolearn_core-1.whl").is_file())
            self.assertTrue(list((mac / SHA[:12]).glob("linux.failed-*/old.whl")))   # moved aside, not deleted
            self.assertTrue((mac / SHA[:12] / "linux-pack.log").is_file())
            self.assertTrue((mac / SHA[:12] / "remote-pack" / "t" / "runner-state.json").is_file())
            self.assertFalse((mac / "state.json").exists())    # the Mac's own state is never overwritten

    def test_failure_lines(self):
        with tempfile.TemporaryDirectory() as td:
            write(Path(td) / "runner.log", "fine\n   STOPPED at linux-pack: audit.sh exited 1\nok\n")
            self.assertIn("STOPPED at linux-pack", RP.failure_lines(td))
            self.assertEqual(RP.failure_lines(Path(td) / "none"), "")


class Collector(unittest.TestCase):
    def receipt(self, out, **kw):
        doc = dict(schema=G.RECORDS_ONLY_SCHEMA, run="99", source_commit=SHA, defines=None, conclusion="success")
        doc.update(kw)
        write(Path(out) / "GHA" / "records-only.json", json.dumps(doc))

    def test_records_only_ok(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(G.records_only_ok(td, SHA)[0])
            self.receipt(td)
            self.assertTrue(G.records_only_ok(td, SHA)[0])
            self.assertFalse(G.records_only_ok(td, BASE_SHA)[0])
            self.assertFalse(G.records_only_ok(td, SHA, "A=1")[0])
            self.receipt(td, defines="A=1")
            self.assertTrue(G.records_only_ok(td, SHA, "A=1")[0])
            self.receipt(td, conclusion="failure")
            self.assertFalse(G.records_only_ok(td, SHA)[0])

    def test_records_only_flag_is_one_set_only(self):
        with tempfile.TemporaryDirectory() as td:
            for extra in (["--host", "--arch", "sm_89"], ["--arch", "sm_89", "--compile-groups", "device"]):
                with self.assertRaises(SystemExit):
                    G.main(["run", "--commit", SHA, "--out", str(Path(td) / "o"), "--records-only", *extra])


class ReleaseModes(unittest.TestCase):
    def ctx(self, pack, version="0.8.67.dev0+0123456789ab"):
        return SimpleNamespace(rel=Path("/w") / SHA[:12], commit=SHA, args=SimpleNamespace(
            candidate_version=version, candidate_pack=pack, candidate_defines="", candidate_ref="",
            candidate_tooling_sha=""))

    def test_candidate_pack(self):
        self.assertEqual(release.candidate_pack(self.ctx("github", version="")), "")   # a release packs locally
        self.assertEqual(release.candidate_pack(self.ctx("github")), "github")
        self.assertEqual(release.candidate_pack(self.ctx("here")), "here")
        self.assertEqual(release.candidate_pack(self.ctx("")), "")

    def test_github_legs(self):
        remote = release.github_legs(self.ctx("github"))
        self.assertTrue(remote and all(type(l) is release.RemoteLeg for l in remote))
        self.assertTrue(all("--records-only" in l.command for l in remote))
        for pack in ("", "here"):
            legs = release.github_legs(self.ctx(pack))
            self.assertTrue(all(type(l) is release.Leg and "--records-only" not in l.command for l in legs))
        # a RemoteLeg is still a Leg (launch_detached's attach and relaunch paths take it)
        self.assertTrue(isinstance(remote[0], release.Leg))

    def test_remote_leg_done_on_its_records(self):
        with tempfile.TemporaryDirectory() as td:
            leg = release.RemoteLeg("cuda-sm_89", "cuda", "sm_89", [], {}, Path(td) / "rb", Path(td), Path(td) / "o")
            write(leg.exit_file, "0")
            self.assertFalse(leg.done(SHA))
            write(Path(td) / "o" / "GHA" / "records-only.json", json.dumps(dict(
                schema=G.RECORDS_ONLY_SCHEMA, run="5", source_commit=SHA, conclusion="success")))
            self.assertTrue(leg.done(SHA))

    def test_the_core_packs_first(self):
        self.assertEqual(next(iter(release.SPLIT_PACKAGES)), "linux")


class Candidate(unittest.TestCase):
    def test_parse_pack(self):
        self.assertIsNone(RC.parse(["--commit", SHA]).pack)
        self.assertEqual(RC.parse(["--commit", SHA, "--pack", "local"]).pack, "local")
        with self.assertRaises(SystemExit):
            RC.parse(["--commit", SHA, "--pack", "box"])

    def test_pack_record(self):
        self.assertEqual(RC.pack_record(None), dict(where="local"))
        self.assertEqual(RC.pack_record({}, here=True), dict(where="github-runner"))
        rec = RC.pack_record(dict(run_id=7, url="u", tag="t", packed_at="2026-11-19T00:00:00+00:00",
                                  conclusion="success", tooling_sha=SHA))
        self.assertEqual((rec["where"], rec["run_id"]), ("github", 7))
        self.assertEqual(RC.pack_record(dict(run_id=7, conclusion=None)), dict(where="local"))

    def test_arm_need_follows_the_pack(self):
        self.assertLess(RC.arm_need(2, "github"), RC.arm_need(2, "local"))
        self.assertEqual(RC.arm_need(2), RC.arm_need(2, "local"))


class Workflows(unittest.TestCase):
    def test_candidate_pack_workflow(self):
        text = (ROOT / ".github" / "workflows" / RP.WORKFLOW).read_text()
        on = text.split("\non:", 1)[1].split("\npermissions:", 1)[0]
        self.assertIn("workflow_dispatch:", on)
        self.assertNotIn("push:", on)
        self.assertNotIn("schedule:", on)
        for key in ("commit", "version", "tooling_sha", "tag", "bundle_url", "bundle_sha256", "base_run"):
            self.assertIn(f"      {key}:", on)
        self.assertIn("${{ inputs.tag }}", text.split("\non:", 1)[0])   # find_run finds it by its tag
        self.assertIn("tools/release_remote_pack.py runner --event \"$GITHUB_EVENT_PATH\"", text)
        self.assertIn(f"name: {RP.ARTIFACT}", text)
        self.assertIn("environments: pkg", text)            # pack_wheel.py's pixi env
        self.assertIn("ref: ${{ inputs.tooling_sha }}", text)
        self.assertNotIn("${{ inputs.bundle_url }}", text)  # read from the event file, masked, never interpolated

    def test_release_linux_build_uploads_the_records_artifact(self):
        text = (ROOT / ".github" / "workflows" / "release-linux-build.yml").read_text()
        assemble = text.split("\n  assemble:", 1)[1].split("\n  host:", 1)[0]
        name = G.RECORDS_ARTIFACT.format(name="${{ matrix.arch == 'gfx942' && 'hip' || 'cuda' }}-${{ matrix.arch }}")
        self.assertIn(f"name: {name}", assemble)
        self.assertIn("path: ${{ runner.temp }}/w/art/leg/GHA", assemble)


if __name__ == "__main__":
    unittest.main()
