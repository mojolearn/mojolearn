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
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto")
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


class AmdRouteTests(unittest.TestCase):
    """The AMD build leg's provider: DigitalOcean, or Hot Aisle when DigitalOcean
    has a GPU droplet live (2026-09-25). No network: the probe is a stand-in,
    and do_gpu_busy is pointed at a local HTTP server."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self._env = os.environ.pop("MOJOLEARN_AMD_PROVIDER", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["MOJOLEARN_AMD_PROVIDER"] = self._env
        self._t.cleanup()

    def ctx(self, busy=None, **kw):
        c = Ctx()
        c.args = args(**kw)
        c.rel = self.tmp
        c.probed = 0
        if busy is not None:
            def probe():
                c.probed += 1
                return busy, "stand-in: " + ("a GPU droplet live" if busy else "none live")
            c.amd_do_probe = probe
        return c

    def amd(self, c):
        (hip,) = [l for l in release.gpu_legs(c) if l.name == "hip-gfx942"]
        return hip

    def test_auto_takes_digitalocean_when_it_is_free(self):
        c = self.ctx(busy=False)
        hip = self.amd(c)
        self.assertEqual(hip.command[:2], ["bash", "tools/do_release061_leg.sh"])
        self.assertEqual(hip.env["MOJOLEARN_RELEASE_UBUNTU22"], "1")
        self.assertEqual(hip.provider, "do")

    def test_auto_takes_hotaisle_when_digitalocean_has_a_gpu_droplet_live(self):
        c = self.ctx(busy=True, amd_expect_from="/nv/leg")
        hip = self.amd(c)
        self.assertEqual(hip.command, ["bash", "tools/hotaisle_release_leg.sh", COMMIT, "--rent", "--expect-from", "/nv/leg"])
        self.assertEqual(hip.env, {"MOJOLEARN_RELEASE_RESULTS_ROOT": str(self.tmp / "legs")})
        # the same tree the packer and linux-wait read, whichever provider built it
        self.assertEqual(hip.release_build, self.tmp / "legs" / "hip-gfx942" / "release-build")
        self.assertEqual(hip.provider, "hotaisle")
        self.assertIn("  AMD build leg: hotaisle (auto: stand-in: a GPU droplet live)", c.lines)
        self.amd(c)
        self.assertEqual(c.probed, 1, "the route is decided once per run")

    def test_pinned_by_flag_or_environment(self):
        c = self.ctx(busy=False, amd_build_provider="hotaisle")
        self.assertEqual(self.amd(c).command[1], "tools/hotaisle_release_leg.sh")
        self.assertEqual(c.probed, 0)
        os.environ["MOJOLEARN_AMD_PROVIDER"] = "hotaisle"
        try:
            c = self.ctx(busy=False)
            self.assertEqual(self.amd(c).command[1], "tools/hotaisle_release_leg.sh")
            c = self.ctx(busy=True, amd_build_provider="do")
            self.assertEqual(self.amd(c).command[1], "tools/do_release061_leg.sh")
        finally:
            del os.environ["MOJOLEARN_AMD_PROVIDER"]

    def test_do_gpu_busy_reads_the_droplet_listing(self):
        import http.server
        import threading
        state = {"droplets": []}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                ok = self.headers.get("Authorization") == "Bearer dop_v1_fake"
                body = json.dumps({"droplets": state["droplets"]} if ok else {"id": "unauthorized"}).encode()
                self.send_response(200 if ok else 401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        api = "http://127.0.0.1:%d/v2" % srv.server_address[1]
        tok = self.tmp / "tok"
        tok.write_text("dop_v1_fake\n")
        try:
            self.assertEqual(release.do_gpu_busy(str(tok), api), (False, "no DigitalOcean GPU droplet live"))
            state["droplets"] = [{"id": 1, "name": "web", "size_slug": "s-1vcpu-1gb"},
                                 {"id": 603, "name": "gpt3-t3", "size_slug": "gpu-mi325x1-256gb"}]
            busy, why = release.do_gpu_busy(str(tok), api)
            self.assertTrue(busy)
            self.assertIn("603:gpt3-t3:gpu-mi325x1-256gb", why)
            self.assertNotIn("web", why)
            tok.write_text("wrong\n")
            self.assertTrue(release.do_gpu_busy(str(tok), api)[0])
            self.assertEqual(release.do_gpu_busy(str(self.tmp / "none"), api)[0], True)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_linux_wait_walks_a_live_droplet_refusal_to_hotaisle(self):
        c = self.ctx(busy=False)
        hip = self.amd(c)
        hip.workdir.mkdir(parents=True, exist_ok=True)
        hip.exit_file.write_text("2\n")
        hip.log.write_text("[amd/rel061] REFUSING: GPU droplet(s) already live, destroy or adopt them first: 603:x:gpu-mi325x1-256gb\n")
        spawned = []
        r = release.Release(args(state_dir=str(self.tmp / "state")),
                            runner=lambda cmd, env, log, detach=False: spawned.append((cmd, env)) or 999999)
        r.state["commit"] = COMMIT
        r.say = lambda msg: None
        saved = release.linux_legs
        release.linux_legs = lambda ctx: [hip]
        try:
            with self.assertRaises(release.StepFailed):   # the relaunched leg is not finished here
                r.step_linux_wait()
        finally:
            release.linux_legs = saved
        self.assertEqual(len(spawned), 1, spawned)
        self.assertIn("tools/hotaisle_release_leg.sh", spawned[0][0][2])
        self.assertEqual(spawned[0][1]["MOJOLEARN_RELEASE_RESULTS_ROOT"], str(hip.workdir))
        self.assertEqual(hip.provider, "hotaisle")
        self.assertEqual(len(list(hip.workdir.glob("hip-gfx942.log.failed-*"))), 1)


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
        env = dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=str(pathlib.Path(self._t.name) / "evidence"),
                   MOJOLEARN_AMD_PROVIDER="do")   # no DigitalOcean probe from a test
        out = subprocess.run([sys.executable, str(ROOT / "tools/release.py"), "0.8.14", "--dry-run",
                              "--state-dir", str(self.state)], capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        for step in release.Release.STEPS:
            self.assertIn("-- " + step, out.stdout)
        # the decision table, in full, and a leg command for every leg it launches
        self.assertIn("== binding reuse plan for", out.stdout)
        self.assertIn("legs to launch:", out.stdout)
        legs = out.stdout.split("legs to launch:", 1)[1].splitlines()[0]
        # the default route is the GPU legs (2026-09-25: no CPU by default)
        self.assertNotIn("tools/release_linux_build.sh", out.stdout)
        if "cuda-" in legs:
            self.assertIn("gemm_remote_leg.sh nvidia", out.stdout)
        if "hip-gfx942" in legs:
            self.assertIn("MOJOLEARN_RELEASE_UBUNTU22=1", out.stdout)
        if not any(n in legs for n in ("cuda-", "hip-")):
            self.assertIn("no build leg", out.stdout)
        self.assertIn("pack-linux-wheel", out.stdout)
        self.assertIn("release_wheel_smoke.sh", out.stdout)
        self.assertIn("--publish", out.stdout)
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
