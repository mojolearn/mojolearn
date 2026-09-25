"""A SEVERABLE release (2026-09-25), proved without a rental: the macOS and
Linux pipelines run at once and each publishes on its own gates; the shipped
source commit is pinned and the tooling runs from a newer checkout; a build
leg of an earlier freeze is taken only when its set identity and build
tooling are unchanged; a column only for a byte-identical wheel; the PASS
ledger (R2 stood in for by a dict); --status. Nothing here reaches a cloud or
the real evidence directory: the runner stands in for every process."""
import argparse
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_ledger  # noqa: E402
import release_tooling  # noqa: E402

X = "a" * 40          # the earlier freeze
Y = "c" * 40          # this freeze
HEAD = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def args(**kw):
    base = dict(version="0.8.99", dry_run=False, publish=None, only="", redo="", build_backend="cpu-box",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                cpu_column=False, status=False, refreeze=False, source_checkout="")
    base.update(kw)
    return argparse.Namespace(**base)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self._env = mock.patch.dict(os.environ, MOJOLEARN_RELEASE_CHECK_DIR=str(self.tmp / "release-check"),
                                    MOJOLEARN_EVIDENCE_ROOT=str(self.tmp / "evidence"),
                                    MOJOLEARN_RELEASE_LEDGER="off")
        self._env.start()
        self.spawned = []

    def tearDown(self):
        self._env.stop()
        self._t.cleanup()

    def release(self, commit=Y, **kw):
        def runner(cmd, env, log, detach=False):
            if detach:
                self.spawned.append((cmd, env))
                return 999999          # not a live pid: the leg reads as finished once its exit file exists
            return 0
        r = release.Release(args(state_dir=str(self.tmp / "state"), **kw), runner=runner)
        r.state["commit"] = commit
        r.lines = []
        r.say = r.lines.append
        r.sleep = lambda s: None
        r.leg_env = lambda: {"MOJOLEARN_SOURCE_CHECKOUT": "/frozen/source"}
        r.packer_takes_legs = lambda: True
        return r


# ---------------------------------------------------------------- (a) two pipelines
class Pipelines(Base):
    """The scheduler with every step stood in for: a step records itself and
    fails where told to."""

    def staged(self, fail=(), gate=None, publish="pypi"):
        r = self.release(publish=publish)
        order, lock = [], threading.Lock()

        def make(step):
            def fn():
                with lock:
                    order.append(step)
                if gate and step in gate:
                    gate[step]()
                if step in fail:
                    raise release.StepFailed("boom " + step)
                if step.startswith("publish-"):
                    return "pypi via alpha-api-tag"
                if step in ("finish-line", "record"):
                    got = r.published_platforms()
                    return f"{step} for {','.join(got)}", dict(platforms=got)
                return "ok"
            return fn
        for step in release.Release.STEPS:
            setattr(r, "step_" + step.replace("-", "_"), make(step))
        return r, order

    def test_linux_failing_leaves_the_mac_publish_done(self):
        r, order = self.staged(fail={"linux-wait"})
        self.assertEqual(r.go(), 1)
        self.assertTrue(r.recorded("publish-macos"))
        self.assertFalse(r.recorded("publish-linux"))
        for later in ("linux-assemble", "linux-pack", "gpu-columns", "publish-linux"):
            self.assertNotIn(later, order)
        self.assertEqual(r.recorded("finish-line")["platforms"], ["macos"])
        self.assertEqual(r.recorded("record")["platforms"], ["macos"])
        text = "\n".join(r.lines)
        self.assertIn("linux: FAILED at linux-wait: boom linux-wait", text)
        self.assertIn("macos: PUBLISHED", text)
        self.assertEqual(r.state["failures"]["linux-wait"]["error"], "boom linux-wait")

    def test_mac_failing_leaves_the_linux_publish_done(self):
        r, order = self.staged(fail={"macos-smoke"})
        self.assertEqual(r.go(), 1)
        self.assertTrue(r.recorded("publish-linux"))
        self.assertFalse(r.recorded("publish-macos"))
        self.assertIn("release-check", order, "the Apple column still runs: the Linux columns are diffed against it")
        self.assertIn("gpu-columns", order)
        self.assertEqual(r.recorded("finish-line")["platforms"], ["linux"])
        text = "\n".join(r.lines)
        self.assertIn("macos: FAILED at macos-smoke", text)
        self.assertIn("linux: PUBLISHED", text)

    def test_a_failed_apple_column_holds_both(self):
        r, order = self.staged(fail={"release-check"})
        self.assertEqual(r.go(), 1)
        self.assertFalse(r.recorded("publish-linux") or r.recorded("publish-macos"))
        self.assertIn("linux-pack", order, "the Linux builds and pack still run")
        self.assertNotIn("gpu-columns", order)
        text = "\n".join(r.lines)
        self.assertIn("linux: FAILED at release-check", text)
        self.assertIn("macos: FAILED at release-check", text)

    def test_linux_waits_never_delay_the_mac_publish(self):
        published = threading.Event()
        gate = {"linux-wait": lambda: self.assertTrue(published.wait(20), "linux-wait held the Mac publish"),
                "publish-macos": lambda: None}
        r, order = self.staged(gate=gate)
        real = r.step_publish_macos

        def pub():
            out = real()
            published.set()
            return out
        r.step_publish_macos = pub
        self.assertEqual(r.go(), 0)
        self.assertLess(order.index("publish-macos"), order.index("linux-assemble"))
        self.assertEqual(sorted(r.recorded("finish-line")["platforms"]), ["linux", "macos"])

    def test_a_rerun_resumes_only_what_is_not_done(self):
        r, _ = self.staged(fail={"gpu-columns"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(r.recorded("record")["platforms"], ["macos"])
        r2, order = self.staged()
        r2.state = r.state
        self.assertEqual(r2.go(), 0)
        self.assertNotIn("publish-macos", order, "a recorded publish is not repeated")
        self.assertNotIn("rehearsal", order)
        self.assertIn("gpu-columns", order)
        # the finish line and the record run again for the newly published platform
        self.assertIn("finish-line", order)
        self.assertEqual(sorted(r2.recorded("record")["platforms"]), ["linux", "macos"])
        self.assertNotIn("failures", {k: v for k, v in r2.state.get("failures", {}).items() if k == "gpu-columns"})

    def test_without_publish_both_pipelines_are_held(self):
        r, order = self.staged(publish=None)
        r.step_publish_linux = lambda: r.publish("linux", None, None)
        r.step_publish_macos = lambda: r.publish("macos", None, None)
        self.assertEqual(r.go(), 1)
        text = "\n".join(r.lines)
        self.assertIn("linux: HELD at publish-linux", text)
        self.assertIn("macos: HELD at publish-macos", text)
        self.assertTrue(r.state["failures"]["publish-linux"]["held"])

    def test_the_pipelines_are_generic(self):
        for name, p in release.PIPELINES.items():
            self.assertEqual(set(p), {"builds", "checks", "publish", "platform"})
            for s in p["builds"] + p["checks"] + [p["publish"]]:
                self.assertIn(s, release.Release.STEPS)
        self.assertEqual(release.RESOURCE["macos-build"], "mac")
        self.assertEqual(release.NEEDS["gpu-columns"], ["linux-pack", "release-check"])
        self.assertEqual(release.NEEDS["publish-macos"], ["macos-smoke", "release-check"])


# ---------------------------------------------------------------- legs
def leg_tree(release_build, vendor, arch, commit, payload=b"x"):
    """A finished leg's release-build tree with a proof naming its one binary."""
    sets = release_build / "build" / "sets" / vendor / arch
    (sets / "host").mkdir(parents=True, exist_ok=True)
    so = sets / "identical" / "_mojolearn_gbdt.so"
    so.parent.mkdir(parents=True, exist_ok=True)
    so.write_bytes(payload)
    (sets / "host" / "_mojolearn_core_host.so").write_bytes(b"host")
    (release_build / "build" / "build-provenance.json").write_text(json.dumps(dict(
        complete=True, build_exit=0, source_commit=commit,
        extensions={f"mojolearn/{vendor}/{arch}/identical/_mojolearn_gbdt.so": sha(payload)})))
    return so


def plan(legs):
    return dict(schema="x", commit=Y, previous=None, rows=[], legs=list(legs), leg_reasons={},
                runtime=dict(decision="BUILD"))


class Legs(Base):
    def cpu_leg(self, r, name):
        (leg,) = [l for l in release.cpu_legs(r) if l.name == name]
        return leg

    def finished(self, leg, commit, rc=0):
        leg_tree(leg.release_build, leg.vendor, leg.arch, commit)
        leg.workdir.mkdir(parents=True, exist_ok=True)
        leg.exit_file.write_text(f"{rc}\n")
        leg.log.write_text("log")

    def test_rerun_after_one_leg_failure_relaunches_only_that_leg(self):
        r = self.release()
        r._plan = plan(["cuda-sm_90a", "cuda-sm_89", "hip-gfx942"])
        for name in ("cuda-sm_90a", "hip-gfx942"):
            self.finished(self.cpu_leg(r, name), Y)
        bad = self.cpu_leg(r, "cuda-sm_89")
        bad.out_dir.mkdir(parents=True)
        bad.workdir.mkdir(parents=True, exist_ok=True)
        bad.exit_file.write_text("1\n")
        bad.log.write_text("boom")
        r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1, self.spawned)
        self.assertIn("--archs sm_89", self.spawned[0][0][2])
        self.assertEqual(len(list(bad.workdir.glob("cuda-sm_89.log.failed-*"))), 1)
        self.assertIn("  cuda-sm_90a: already done at this freeze", r.lines)
        # the relaunched leg names both commits
        prov = json.loads(bad.provenance_file.read_text())
        self.assertEqual(prov["source_commit"], Y)
        self.assertEqual(prov["tooling_commit"], HEAD)
        self.assertIn("tooling_digest", prov)
        self.assertIn("--archs", prov["command"])

    def setup_earlier_freeze(self, name, vendor, arch, digest, tooling="T" * 64, payload=b"x"):
        old = self.tmp / "state" / X[:12] / "legs"
        rb, out = release.leg_layout(old, "cpu-box", name)
        leg_tree(rb, vendor, arch, X, payload)
        old.mkdir(parents=True, exist_ok=True)
        (old / f"{name}.exit").write_text("0\n")
        (old / f"{name}.provenance.json").write_text(json.dumps(dict(
            leg=name, vendor=vendor, arch=arch, source_commit=X, tooling_commit="1" * 40, tooling_digest=tooling,
            overlay=None, set_identity=dict(digest=digest), release_build=str(rb), out_dir=str(out),
            exit_file=str(old / f"{name}.exit"))))
        return rb

    def identities(self, r, table, tooling="T" * 64):
        r.set_identity = lambda vendor, arch: {"digest": table[f"{vendor}-{arch}"]} if f"{vendor}-{arch}" in table else None
        r.tooling = lambda: dict(commit=HEAD, digest=tooling, dirty=[], files={})

    def test_an_unchanged_set_takes_its_completed_leg_and_a_changed_one_rebuilds(self):
        self.setup_earlier_freeze("cuda-sm_90a", "cuda", "sm_90a", "D90")
        self.setup_earlier_freeze("hip-gfx942", "hip", "gfx942", "DHIP-OLD")
        r = self.release()
        r._plan = plan(["cuda-sm_90a", "hip-gfx942"])
        self.identities(r, {"cuda-sm_90a": "D90", "hip-gfx942": "DHIP-NEW"})
        result = r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1)
        self.assertIn("--archs gfx942", self.spawned[0][0][2])
        self.assertIn("taken from an earlier freeze: cuda-sm_90a", result)
        doc = json.loads(r.admission_path("cuda-sm_90a").read_text())
        self.assertEqual((doc["built_from"], doc["admitted_for"]), (X, Y))
        self.assertIn("set identity", doc["because"])
        self.assertEqual(doc["set_identity_digest"], "D90")
        self.assertTrue(any("TAKEN, not launched: built from aaaaaaaaaaaa" in l for l in r.lines), r.lines)
        self.assertTrue(any("hip-gfx942" in l and "set identity differs" in l for l in r.lines), r.lines)
        self.assertEqual(set(r.admitted_legs()), {"cuda-sm_90a"})
        # linux-wait does not wait for a taken leg, and the pack gives no proof for it
        self.assertEqual([l.name for l in r.reused_leg_objects()], ["cuda-sm_90a"])
        origin = r.leg_origin(doc)
        for k in ("source_commit", "leg", "proof_sha256", "admitted_for", "set_identity_digest", "tooling_digest"):
            self.assertTrue(origin[k], k)

    def test_changed_tooling_or_moved_bytes_rebuild(self):
        rb = self.setup_earlier_freeze("cuda-sm_90a", "cuda", "sm_90a", "D90")
        r = self.release()
        r._plan = plan(["cuda-sm_90a"])
        self.identities(r, {"cuda-sm_90a": "D90"}, tooling="U" * 64)
        r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1, "different build tooling: rebuilt")
        self.assertTrue(any("build tooling differs" in l for l in r.lines), r.lines)

        self.spawned.clear()
        r = self.release()
        r._plan = plan(["cuda-sm_90a"])
        self.identities(r, {"cuda-sm_90a": "D90"})
        (rb / "build" / "sets" / "cuda" / "sm_90a" / "identical" / "_mojolearn_gbdt.so").write_bytes(b"moved")
        r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1, "a byte that is not the proof's: rebuilt")
        self.assertTrue(any("not the proof's bytes" in l for l in r.lines), r.lines)

        self.spawned.clear()
        r = self.release()
        r._plan = plan(["cuda-sm_90a"])
        self.identities(r, {"cuda-sm_90a": "D90"})
        r.tooling = lambda: dict(commit=HEAD, digest=None, dirty=["tools/gemm_remote_leg.sh"], files={})
        r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1, "uncommitted tooling: nothing is taken")

    def test_a_source_whose_packer_predates_leg_origins_rebuilds(self):
        self.setup_earlier_freeze("cuda-sm_90a", "cuda", "sm_90a", "D90")
        r = self.release()
        r._plan = plan(["cuda-sm_90a"])
        self.identities(r, {"cuda-sm_90a": "D90"})
        del r.packer_takes_legs
        with mock.patch.object(release, "file_at", return_value="def load_reuse(): pass\n"):
            r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1)
        self.assertTrue(any("packer predates" in l for l in r.lines), r.lines)
        with mock.patch.object(release, "file_at", return_value="LEG_ORIGIN_KEYS = ()\n"):
            self.assertTrue(r.packer_takes_legs())

    def test_an_unreadable_set_identity_rebuilds(self):
        self.setup_earlier_freeze("cuda-sm_90a", "cuda", "sm_90a", "D90")
        r = self.release()
        r._plan = plan(["cuda-sm_90a"])
        self.identities(r, {})
        r.step_linux_builds()
        self.assertEqual(len(self.spawned), 1)

    def test_the_ledger_finds_a_leg_of_another_version_and_records_a_built_one(self):
        # the leg lives outside this version's directory; only the ledger knows it
        rb = self.tmp / "elsewhere" / "cuda-sm_89" / "cuda-sm_89" / "release-build"
        leg_tree(rb, "cuda", "sm_89", X)
        (self.tmp / "elsewhere" / "cuda-sm_89.exit").write_text("0\n")
        r = self.release()
        r._plan = plan(["cuda-sm_89"])
        self.identities(r, {"cuda-sm_89": "D89"})
        r.ledger().put(release_ledger.build_key("cuda", "sm_89", "D89", "T" * 64), dict(
            verdict="PASS", provenance=dict(leg="cuda-sm_89", source_commit=X, tooling_digest="T" * 64,
                                            set_identity=dict(digest="D89"), release_build=str(rb),
                                            out_dir=str(rb.parent), exit_file=str(self.tmp / "elsewhere" / "cuda-sm_89.exit"))))
        r.step_linux_builds()
        self.assertEqual(self.spawned, [])
        self.assertIn("ledger build/cuda-sm_89/D89/", r.admitted_leg("cuda-sm_89")["provenance"])

        # a leg built at this freeze goes into the ledger keyed by what it built, with both commits
        leg = self.cpu_leg(r, "cuda-sm_90a")
        self.finished(leg, Y)
        leg.provenance_file.write_text(json.dumps(dict(set_identity=dict(digest="D90"), tooling_digest="T" * 64,
                                                       tooling_commit=HEAD, source_commit=Y)))
        r.record_leg(leg)
        (entry,) = r.ledger().find("build/cuda-sm_90a/D90/")
        self.assertEqual((entry["source_commit"], entry["tooling_commit"], entry["verdict"]), (Y, HEAD, "PASS"))


# ---------------------------------------------------------------- (d) columns
def column(vendor, h="aaaa"):
    return dict(mode="identical", repeats=1, vendor=vendor, commit=Y, complete=True,
                cells={"rf-clf/base": {"verdict": "STABLE", "hashes": [h], "parts": [{"predict": h}]}})


class Columns(Base):
    def setup_release(self, wheel_bytes=b"payload"):
        r = self.release()
        final = r.rel / "linux" / "final" / "mojolearn-0.8.99-py3-none-manylinux_2_35_x86_64.whl"
        final.parent.mkdir(parents=True)
        with zipfile.ZipFile(final, "w") as z:
            z.writestr("mojolearn/identity_columns/COMMIT", Y + "\n")
            z.writestr("x", wheel_bytes)
        ref = self.tmp / "release-check" / Y[:12] / "metal"
        ref.mkdir(parents=True)
        (ref / "column.json").write_text(json.dumps(column("apple-m4")))
        (ref / "run-summary.json").write_text(json.dumps(dict(complete=True, validation_failures=[])))

        def selection(vendor):
            p = r.rel / f"selection-{vendor}.json"
            p.write_text(json.dumps(dict(backend=vendor, column=vendor, fixtures="base", lanes=["rf-clf"], commit=Y)))
            return p
        r.gpu_selection = selection
        # NVIDIA is PASSED for this wheel already; the AMD column is the one under test
        out = r.rel / "smoke-linux"
        out.mkdir(parents=True)
        (out / "results.json").write_text(json.dumps(dict(status="PASSED", scope="expanded", source_commit=Y,
                                                          wheel_sha256=hashlib.sha256(final.read_bytes()).hexdigest())))
        (out / "column-cuda.json").write_text(json.dumps(column("cuda")))
        (out / "diff-ref-cuda.txt").write_text("summary: IDENTICAL=1\n")
        return r, final

    def earlier_amd(self, wheel_sha, lanes=("rf-clf",)):
        d = self.tmp / "state" / X[:12] / "column-amd"
        d.mkdir(parents=True)
        (d / "column-hip.json").write_text(json.dumps(column("hip")))
        sel = dict(backend="hip", column="hip", fixtures="base", lanes=list(lanes), commit=X)
        seld = hashlib.sha256(json.dumps({k: sel[k] for k in ("backend", "column", "fixtures", "lanes")},
                                         sort_keys=True).encode()).hexdigest()
        (d / "column-provenance.json").write_text(json.dumps(dict(
            vendor="hip", wheel_sha256=wheel_sha, selection_digest=seld, source_commit=X,
            column_sha256=hashlib.sha256((d / "column-hip.json").read_bytes()).hexdigest())))
        return d

    def test_a_column_is_taken_for_a_byte_identical_wheel(self):
        r, final = self.setup_release()
        self.earlier_amd(release.sha256(final))
        result = r.step_gpu_columns()
        self.assertEqual(self.spawned, [], "nothing launched")
        out = r.rel / "column-amd"
        self.assertTrue((out / "reused.json").is_file())
        self.assertIn("summary", (out / "diff-ref-hip.txt").read_text(), "diffed again against this release's Apple column")
        doc = json.loads((out / "reused.json").read_text())
        self.assertEqual(doc["admitted_for"], Y)
        self.assertIn("byte-identical", doc["because"])
        self.assertIn("no DIVERGENT cell across the columns", result)

    def test_a_column_of_another_wheel_or_other_lanes_is_not_taken(self):
        r, final = self.setup_release()
        self.earlier_amd("0" * 64)
        with self.assertRaises(release.StepFailed):     # the stand-in launch writes no column
            r.step_gpu_columns()
        self.assertEqual(len(self.spawned), 1)
        self.assertIn("--vendor hip", self.spawned[0][0][2])
        self.assertFalse((r.rel / "column-amd" / "reused.json").exists())

        self.spawned.clear()
        self._t2 = tempfile.TemporaryDirectory()
        self.addCleanup(self._t2.cleanup)
        self.tmp = pathlib.Path(self._t2.name)
        with mock.patch.dict(os.environ, MOJOLEARN_RELEASE_CHECK_DIR=str(self.tmp / "release-check")):
            r, final = self.setup_release()
            self.earlier_amd(release.sha256(final), lanes=("rf-clf", "knn"))
            with self.assertRaises(release.StepFailed):
                r.step_gpu_columns()
        self.assertEqual(len(self.spawned), 1, "another lane selection: launched")


# ---------------------------------------------------------------- (e) --status
class Status(Base):
    def test_status_reports_and_writes_nothing(self):
        r = self.release()
        r.state["steps"] = {s: dict(done=True, commit=Y, at="2026-09-25T10:00:00Z", result="ok")
                            for s in ("freeze-version", "freeze-changelog", "freeze-docs-facts", "freeze-commit",
                                      "rehearsal", "reuse-plan", "linux-builds", "macos-build")}
        r.state["failures"] = {"macos-smoke": dict(at="2026-09-25T11:00:00Z", commit=Y, error="smoke exited 1; log L",
                                                   held=False)}
        r.save()
        release.write_json(r.plan_path, plan(["cuda-sm_90a", "cuda-sm_89", "hip-gfx942"]))
        legs = r.rel / "legs"
        rb, _ = release.leg_layout(legs, "cpu-box", "cuda-sm_90a")
        leg_tree(rb, "cuda", "sm_90a", Y)
        (legs / "cuda-sm_90a.exit").write_text("0\n")
        (legs / "cuda-sm_89.exit").write_text("1\n")
        (legs / "cuda-sm_89.log").write_text("boom")
        (legs / "hip-gfx942.pid").write_text(str(os.getpid()))
        before = sorted((p.relative_to(self.tmp).as_posix(), p.stat().st_mtime_ns) for p in self.tmp.rglob("*"))
        st = release.Release(args(state_dir=str(self.tmp / "state"), status=True))
        st.lines = []
        st.say = st.lines.append
        self.assertEqual(st.go(), 0)
        after = sorted((p.relative_to(self.tmp).as_posix(), p.stat().st_mtime_ns) for p in self.tmp.rglob("*"))
        self.assertEqual(before, after, "--status wrote something")
        text = "\n".join(st.lines)
        self.assertIn("source commit " + Y, text)
        for title in ("-- common", "-- macos pipeline", "-- linux pipeline", "-- finish"):
            self.assertIn(title, text)
        self.assertRegex(text, r"macos-smoke +failed +2026-09-25T11:00:00Z: smoke exited 1; log L")
        self.assertIn("rerun: run it again", text)
        self.assertRegex(text, r"cuda-sm_90a +done +exit 0, proof of cccccccccccc")
        self.assertRegex(text, r"cuda-sm_89 +failed +exit 1, log .*legs/cuda-sm_89.log")
        self.assertIn("rerun: relaunch it", text)
        self.assertRegex(text, r"hip-gfx942 +running +pid ")
        self.assertRegex(text, r"nvidia +owed +no final Linux wheel yet")
        self.assertIn("== macos: failed at macos-smoke, linux: owed", text)
        self.assertIn("a rerun would:", text)

    def test_the_flag_parses_alone(self):
        with mock.patch.object(release.Release, "go", lambda self: self.args):
            self.assertTrue(release.main(["0.8.99", "--status"]).status)
            self.assertTrue(release.main(["0.8.99", "--refreeze"]).refreeze)
        with self.assertRaises(SystemExit):
            release.main(["0.8.99", "--status", "--dry-run"])


# ---------------------------------------------------------------- source and tooling
class SourceAndTooling(Base):
    def test_a_tooling_commit_after_freeze_does_not_refreeze_or_discard_legs(self):
        r = self.release(commit=X)
        r.state["steps"]["linux-builds"] = dict(done=True, commit=X, at="t", result="3 legs")
        leg = release.cpu_legs(r)[0]
        leg_tree(leg.release_build, leg.vendor, leg.arch, X)
        leg.workdir.mkdir(parents=True, exist_ok=True)
        leg.exit_file.write_text("0\n")
        with mock.patch.object(release, "git", return_value="b" * 40):
            result = r.step_freeze_commit()
        self.assertIn("source pinned at " + X, result)
        self.assertIn("tooling runs from HEAD bbbbbbbbbbbb", result)
        self.assertEqual(r.state["commit"], X)
        self.assertTrue(r.recorded("linux-builds"))
        self.assertEqual(release.launch_detached(r, [leg]), 0)
        self.assertEqual(self.spawned, [])

    def test_frozen_source_is_checked_not_edited(self):
        version = release.re.search(r'__version__ = "([^"]*)"',
                                    (ROOT / "python/mojolearn/_version.py").read_text()).group(1)
        r = release.Release(args(state_dir=str(self.tmp / "state"), version=version))
        r.state["commit"] = HEAD
        r.say = lambda m: None
        self.assertIn(f"says {version}", r.step_freeze_version())
        self.assertIn("frozen source", r.step_freeze_changelog())
        self.assertIn("part of the frozen source", r.step_freeze_docs_facts())
        r.args.version = r.version = "9.9.9"
        with self.assertRaises(release.StepFailed):
            r.step_freeze_version()

    def test_refreeze_is_refused_once_a_wheel_is_published(self):
        r = self.release(commit=X, refreeze=True)
        r.state["steps"]["publish-macos"] = dict(done=True, commit=X, at="t", result="pypi")
        with mock.patch.object(release, "git", return_value="b" * 40):
            with self.assertRaises(release.StepFailed) as cm:
                r.step_freeze_commit()
        self.assertIn("already published", str(cm.exception))

    def test_publish_ships_the_frozen_source_from_the_tooling_checkout(self):
        r = self.release(commit=Y, publish="pypi")
        calls = []
        r.runner = lambda cmd, env, log, detach=False: calls.append((cmd, env)) or 0
        whl = r.rel / "macos" / "mojolearn-0.8.99-py3-none-macosx_11_0_arm64.whl"
        whl.parent.mkdir(parents=True)
        with zipfile.ZipFile(whl, "w") as z:
            z.writestr("mojolearn/identity_columns/COMMIT", Y + "\n")
        r.on_pypi = lambda w: False
        out = r.step_publish_macos()
        (cmd, env), = calls
        self.assertEqual(cmd[:2], ["bash", "tools/release_linux_publish.sh"])
        self.assertEqual(env["MOJOLEARN_ARTIFACT_SOURCE_COMMIT"], Y)
        self.assertIn(f"source {Y[:12]}, tooling {HEAD[:12]}", out)
        with zipfile.ZipFile(whl, "w") as z:
            z.writestr("mojolearn/identity_columns/COMMIT", X + "\n")
        with self.assertRaises(release.StepFailed):
            r.step_publish_macos()

    def test_the_overlay_refuses_source_inventory_files(self):
        for rel in ("bindings/build.sh", "k.mojo", "packaging/linux/build_sets.sh", "python/mojolearn/_backend.py",
                    "tokenizer/tools/gen.py", "pixi.lock", "pixi.toml", "tools/linux_surface_qualification.sh",
                    "../etc/passwd"):
            with self.assertRaises(release_tooling.OverlayRefused, msg=rel):
                release_tooling.overlay_manifest(ROOT, HEAD, HEAD, files=[rel])
        for rel in release_tooling.BOX_OVERLAY:
            self.assertIsNone(release_tooling.refused(rel), rel)
        self.assertEqual(release_tooling.overlay_manifest(ROOT, HEAD, HEAD)["files"], {})

    def test_the_overlay_carries_only_changed_tooling_bytes(self):
        repo = self.tmp / "repo"
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for rel in release_tooling.BOX_OVERLAY:
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text("v1 " + rel + "\n")
        subprocess.run(["git", "init", "-q", str(repo)], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "source"], check=True, env=env)
        src = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (repo / "tools/amd_serial_guard.py").write_text("v2 guard\n")
        subprocess.run(["git", "-C", str(repo), "commit", "-qam", "tooling fix"], check=True, env=env)
        tool = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (repo / "tools/amd_serial_guard.py").write_text("uncommitted\n")    # never shipped
        m = release_tooling.overlay_manifest(repo, tool, src)
        self.assertEqual(list(m["files"]), ["tools/amd_serial_guard.py"])
        self.assertEqual(m["files"]["tools/amd_serial_guard.py"]["tooling_sha256"], sha(b"v2 guard\n"))
        self.assertEqual(release_tooling.effective_builders(m), {})
        tgz, digest = release_tooling.write_overlay(repo, m, self.tmp / "ov")
        self.assertEqual(digest, release.sha256(tgz))
        with tarfile.open(tgz) as t:
            self.assertEqual(t.getnames(), ["tools/amd_serial_guard.py"])
            self.assertEqual(t.extractfile("tools/amd_serial_guard.py").read(), b"v2 guard\n")
        again, digest2 = release_tooling.write_overlay(repo, m, self.tmp / "ov2")
        self.assertEqual(digest, digest2, "the same overlay makes the same bytes")
        # a builder in the overlay enters the binding identities with the bytes that run
        (repo / "tools/release061_remote_build.sh").write_text("v2 build\n")
        subprocess.run(["git", "-C", str(repo), "commit", "-qam", "builder fix"], check=True, env=env)
        tool2 = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        m2 = release_tooling.overlay_manifest(repo, tool2, src)
        self.assertEqual(release_tooling.effective_builders(m2),
                         {"tools/release061_remote_build.sh": sha(b"v2 build\n")})

    def test_the_box_applies_and_verifies_the_overlay(self):
        """tools/route_overlay_lib.sh end to end, with this machine as the box."""
        box = self.tmp / "box" / "mojolearn"
        (box / "tools").mkdir(parents=True)
        (box / "tools/amd_serial_guard.py").write_text("source copy\n")

        def tarball(members):
            raw = io.BytesIO()
            with tarfile.open(fileobj=raw, mode="w:gz") as t:
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
            p = self.tmp / f"ov{len(list(self.tmp.glob('ov*.tgz')))}.tgz"
            p.write_bytes(raw.getvalue())
            return p
        good = tarball({"tools/amd_serial_guard.py": b"tooling copy\n"})
        lib = ROOT / "tools" / "route_overlay_lib.sh"
        script = (f'. "{lib}"; ro_prepare "$1" "$2" "$3" || exit 3; '
                  f'bash -c "$(ro_remote_cmd "$4")" < "$1" > "$3/route-overlay.txt" || exit 4; '
                  f'ro_verify "$3/route-overlay.txt" || exit 5; cat "$3/route-overlay.txt"')
        scratch = self.tmp / "scratch"
        scratch.mkdir()
        p = subprocess.run(["bash", "-c", script, "x", str(good), release.sha256(good), str(scratch), str(box)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual((box / "tools/amd_serial_guard.py").read_text(), "tooling copy\n")
        self.assertIn("before tools/amd_serial_guard.py " + sha(b"source copy\n"), p.stdout)
        self.assertIn("after tools/amd_serial_guard.py " + sha(b"tooling copy\n"), p.stdout)
        self.assertIn("overlay_sha256=" + release.sha256(good), p.stdout)
        for bad, why in ((tarball({"bindings/build_x.sh": b"x"}), "source inventory"),
                         (tarball({"core/k.mojo": b"x"}), "source inventory"),
                         (tarball({"python/mojolearn/a.py": b"x"}), "source inventory")):
            p = subprocess.run(["bash", "-c", script, "x", str(bad), release.sha256(bad), str(scratch), str(box)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 3, p.stdout)
            self.assertIn(why, p.stdout)
        p = subprocess.run(["bash", "-c", script, "x", str(good), "0" * 64, str(scratch), str(box)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 3, "a tarball that is not the recorded one is refused")

    def test_linux_wait_refuses_a_leg_whose_overlay_was_not_applied(self):
        r = self.release()
        r._plan = plan(["cuda-sm_89"])
        (leg,) = [l for l in release.cpu_legs(r) if l.name == "cuda-sm_89"]
        leg_tree(leg.release_build, "cuda", "sm_89", Y)
        leg.workdir.mkdir(parents=True, exist_ok=True)
        leg.exit_file.write_text("0\n")
        leg.pid_file.write_text("999999")
        leg.provenance_file.write_text(json.dumps(dict(overlay=dict(files={
            "tools/amd_serial_guard.py": dict(tooling_sha256="f" * 64)}))))
        with self.assertRaises(release.StepFailed) as cm:
            r.step_linux_wait()
        self.assertIn("route-overlay.txt", str(cm.exception))
        leg.out_dir.mkdir(parents=True, exist_ok=True)
        (leg.out_dir / "route-overlay.txt").write_text("after tools/amd_serial_guard.py " + "f" * 64 + "\n")
        self.assertIn("all legs built", r.step_linux_wait())


# ---------------------------------------------------------------- the ledger
class FakeR2:
    def __init__(self):
        self.objects, self.fail = {}, False

    def _check(self):
        if self.fail:
            raise OSError("R2 is down")

    def get(self, key):
        self._check()
        return self.objects.get(key)

    def put(self, key, body):
        self._check()
        self.objects[key] = body

    def list(self, prefix):
        self._check()
        return [k for k in self.objects if k.startswith(prefix)]


class Ledger(unittest.TestCase):
    def test_once_remote_mirror_and_outage(self):
        with tempfile.TemporaryDirectory() as d:
            r2 = FakeR2()
            led = release_ledger.Ledger(pathlib.Path(d) / "a", r2)
            key = release_ledger.column_key("hip", "w" * 64, "s" * 64)
            led.put(key, dict(verdict="PASS", source_commit=X, tooling_commit=HEAD))
            self.assertIn(release_ledger.PREFIX + key, r2.objects)
            led.put(key, dict(verdict="PASS", source_commit=Y))
            self.assertEqual(json.loads(r2.objects[release_ledger.PREFIX + key])["source_commit"], X, "recorded once")
            # another machine: only R2 knows it; a read fills its mirror
            other = release_ledger.Ledger(pathlib.Path(d) / "b", r2)
            self.assertEqual(other.get(key)["source_commit"], X)
            self.assertTrue((pathlib.Path(d) / "b" / key).is_file())
            self.assertEqual([e["key"] for e in other.find("column/hip/")], [key])
            # an outage never raises; it costs a rerun
            r2.fail = True
            third = release_ledger.Ledger(pathlib.Path(d) / "c", r2)
            self.assertIsNone(third.get(key))
            third.put(release_ledger.apple_key(Y), dict(verdict="PASS"))
            self.assertTrue(third.errors)

    def test_credentials_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "r2"
            p.write_text("export R2_ACCOUNT_ID=acct\nR2_ACCESS_KEY_ID='key'\nR2_SECRET_ACCESS_KEY=\"s\"\nR2_BUCKET=mojolearn-data\n")
            self.assertEqual(release_ledger.read_creds(p)["R2_BUCKET"], "mojolearn-data")
            p.write_text("R2_BUCKET=x\n")
            self.assertIsNone(release_ledger.read_creds(p))

    def test_tests_and_dry_runs_never_reach_r2(self):
        with tempfile.TemporaryDirectory() as d:
            r = release.Release(args(state_dir=d, dry_run=True))
            self.assertIsNone(r.ledger().client)
            with mock.patch.dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=d):
                r = release.Release(args(), runner=lambda *a, **k: 0)
                self.assertIsNone(r.ledger().client)


if __name__ == "__main__":
    unittest.main()
