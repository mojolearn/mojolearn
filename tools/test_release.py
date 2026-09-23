"""tools/release.py without running a release: the freeze helpers, the leg
launcher's resume rules, the cross-leg host digest check, receipt matching, the
publish gate and the dry run. No command it plans is executed."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402

COMMIT = "a" * 40


def args(**kw):
    base = dict(version="0.8.14", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir="")
    base.update(kw)
    return argparse.Namespace(**base)


def wheel(path, commit=COMMIT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mojolearn/identity_columns/COMMIT", commit + "\n")
    return path


class Ctx:
    """The slice of Release that launch_linux_builds uses."""

    def __init__(self, commit=COMMIT):
        self.commit, self.spawned, self.slept, self.lines = commit, [], 0, []

    def say(self, msg):
        self.lines.append(msg)

    def sleep(self, s):
        self.slept += s

    def spawn(self, cmd, env):
        self.spawned.append((cmd, env))
        return 999999  # not a live pid


def leg(tmp, name="cuda-sm_89", vendor="cuda", arch="sm_89"):
    out = tmp / "legs" / name
    return release.Leg(name, vendor, arch, ["true"], {"X": "1"}, out / "remote" / "release-build",
                       tmp / "legs", out)


def build_tree(lg, commit=COMMIT, host=b"same"):
    sets = lg.release_build / "build" / "sets" / lg.vendor / lg.arch / "host"
    sets.mkdir(parents=True, exist_ok=True)
    (sets / "_mojolearn_core_host.so").write_bytes(host)
    (lg.release_build / "build" / "build-provenance.json").write_text(
        json.dumps(dict(complete=True, build_exit=0, source_commit=commit)))


class FreezeTests(unittest.TestCase):
    def test_set_version_and_changelog(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "python/mojolearn").mkdir(parents=True)
            (root / "python/mojolearn/_version.py").write_text('# x\n__version__ = "0.8.14"\n')
            (root / "python/pyproject.toml").write_text('[project]\nname = "m"\nversion = "0.8.14"\n')
            self.assertEqual(release.set_version(root, "0.8.15"),
                             ["python/mojolearn/_version.py", "python/pyproject.toml"])
            self.assertEqual(release.set_version(root, "0.8.15"), [])
            self.assertIn('version = "0.8.15"', (root / "python/pyproject.toml").read_text())
            (root / "CHANGELOG.md").write_text("# C\n\n## 0.8.15 (published 2026-09-23)\n\n## 0.8.14 (published 2026-09-22)\n")
            self.assertEqual(release.changelog_date(root, "0.8.15"), "2026-09-23")
            self.assertIsNone(release.changelog_date(root, "0.8.16"))

    def test_docs_fact_files_come_from_docs_facts(self):
        self.assertIn("README.md", release.docs_fact_files())


class LegTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_launch_staggers_and_detaches(self):
        ctx = Ctx()
        legs = [leg(self.tmp), leg(self.tmp, "cuda-sm_90a", "cuda", "sm_90a")]
        self.assertEqual(release.launch_linux_builds(ctx, legs), 2)
        self.assertEqual(ctx.slept, 90)
        self.assertTrue(all(env["X"] == "1" for _, env in ctx.spawned))
        self.assertIn("echo $? >", ctx.spawned[0][0][2])
        self.assertEqual(legs[0].pid(), 999999)

    def test_built_leg_is_not_relaunched_and_failed_leg_moves_aside(self):
        ctx = Ctx()
        good, bad = leg(self.tmp), leg(self.tmp, "hip-gfx942", "hip", "gfx942")
        build_tree(good)
        good.workdir.mkdir(parents=True, exist_ok=True)
        good.exit_file.write_text("0\n")
        bad.out_dir.mkdir(parents=True)
        bad.exit_file.write_text("10\n")
        bad.log.write_text("boom")
        self.assertEqual(release.launch_linux_builds(ctx, [good, bad]), 1)
        self.assertEqual(len(ctx.spawned), 1)
        self.assertFalse(bad.out_dir.exists())
        self.assertEqual(len(list(self.tmp.glob("legs/hip-gfx942.failed-*"))), 1)
        self.assertEqual(len(list(self.tmp.glob("legs/hip-gfx942.log.failed-*"))), 1)

    def test_proof_must_name_this_commit(self):
        lg = leg(self.tmp)
        build_tree(lg, commit="b" * 40)
        self.assertFalse(lg.proof_ok(COMMIT))
        self.assertTrue(lg.proof_ok("b" * 40))

    def test_host_digest_mismatch_is_named(self):
        a, b = leg(self.tmp), leg(self.tmp, "hip-gfx942", "hip", "gfx942")
        build_tree(a, host=b"nvidia")
        build_tree(b, host=b"amd")
        bad = release.host_digest_mismatches([a, b])
        self.assertEqual(set(bad), {"_mojolearn_core_host.so"})
        self.assertEqual(set(bad["_mojolearn_core_host.so"]), {"cuda-sm_89", "hip-gfx942"})
        build_tree(b, host=b"nvidia")
        self.assertEqual(release.host_digest_mismatches([a, b]), {})


class ReceiptTests(unittest.TestCase):
    def test_smoke_receipt_must_be_about_this_wheel(self):
        with tempfile.TemporaryDirectory() as d:
            w = wheel(pathlib.Path(d) / "mojolearn-0.8.14-py3-none-macosx_11_0_arm64.whl")
            digest = hashlib.sha256(w.read_bytes()).hexdigest()
            r = pathlib.Path(d) / "results.json"
            r.write_text(json.dumps(dict(status="PASSED", scope="expanded", wheel_sha256=digest,
                                         source_commit=COMMIT)))
            self.assertTrue(release.smoke_passed(r, w))
            r.write_text(json.dumps(dict(status="PASSED", scope="expanded", wheel_sha256="0" * 64,
                                         source_commit=COMMIT)))
            self.assertFalse(release.smoke_passed(r, w))
            self.assertFalse(release.smoke_passed(pathlib.Path(d) / "missing.json", w))


class RunTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.state = pathlib.Path(self._t.name) / "state"

    def tearDown(self):
        self._t.cleanup()

    def release(self, **kw):
        calls = []
        r = release.Release(args(state_dir=str(self.state), **kw),
                            runner=lambda cmd, env, log, detach=False: calls.append(cmd) or 0)
        r.state["commit"] = COMMIT
        r.say = lambda msg: None
        return r, calls

    def test_publish_needs_an_explicit_flag(self):
        r, calls = self.release(only="publish-linux")
        self.assertEqual(r.go(), 1)
        self.assertEqual(calls, [])

    def test_done_smoke_is_not_rerun(self):
        r, calls = self.release(only="macos-smoke")
        w = wheel(r.rel / "macos" / "mojolearn-0.8.14-py3-none-macosx_11_0_arm64.whl")
        (r.rel / "smoke-macos").mkdir(parents=True)
        (r.rel / "smoke-macos" / "results.json").write_text(json.dumps(dict(
            status="PASSED", scope="expanded", wheel_sha256=hashlib.sha256(w.read_bytes()).hexdigest(),
            source_commit=COMMIT)))
        self.assertEqual(r.go(), 0)
        self.assertEqual(calls, [])
        self.assertTrue(r.recorded("macos-smoke"))

    def test_recorded_publish_is_skipped(self):
        r, calls = self.release(only="publish-linux", publish="pypi")
        r.mark("publish-linux", result="pypi via alpha-api-0.8.14-linux-20260922")
        self.assertEqual(r.go(), 0)
        self.assertEqual(calls, [])

    def test_dry_run_plans_every_step_and_writes_nothing(self):
        # The identity cache lives under the evidence root; the dry run must
        # write nothing under the state directory.
        env = dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=str(pathlib.Path(self._t.name) / "evidence"))
        out = subprocess.run([sys.executable, str(ROOT / "tools/release.py"), "0.8.14", "--dry-run",
                              "--state-dir", str(self.state)], capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        for step in release.Release.STEPS:
            self.assertIn("-- " + step, out.stdout)
        # the decision table, in full, and a leg command for every leg it launches
        self.assertIn("== binding reuse plan for", out.stdout)
        self.assertIn("legs to launch:", out.stdout)
        legs = out.stdout.split("legs to launch:", 1)[1].splitlines()[0]
        if "cuda-" in legs:
            self.assertIn("gemm_remote_leg.sh nvidia", out.stdout)
        else:
            self.assertNotIn("gemm_remote_leg.sh nvidia", out.stdout)
        if "hip-gfx942" in legs:
            self.assertIn("MOJOLEARN_RELEASE_UBUNTU22=1", out.stdout)
        else:
            self.assertIn("no build leg", out.stdout)
        self.assertIn("pack-linux-wheel", out.stdout)
        self.assertIn("release_wheel_smoke.sh", out.stdout)
        self.assertIn("--publish", out.stdout)
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
