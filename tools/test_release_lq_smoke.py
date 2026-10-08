"""tools/release_lq_smoke.py and tools/release.py --smoke-via lq (2026-10-08:
releases rent nothing). Every subprocess is mocked: no ssh, no lq, no box."""
import argparse
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_lq_smoke as lqs  # noqa: E402

COMMIT = "a" * 40
LQ_TEXT = '''#!/bin/bash
PEM=$HOME/.ssh/x.pem; SO="-o ConnectTimeout=15"
NV="-p 30401 root@195.0.2.1"; AMD="root@192.0.2.2"
'''


def tgz(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Plan(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self.lq = self.tmp / "lq"
        self.lq.write_text(LQ_TEXT)

    def tearDown(self):
        self._t.cleanup()

    def test_targets_are_lqs_own(self):
        self.assertEqual(lqs.lq_target(self.lq, "nv"), ["-p", "30401", "root@195.0.2.1"])
        self.assertEqual(lqs.lq_target(self.lq, "amd"), ["root@192.0.2.2"])

    def test_box_plan_renames_files_and_drops_the_rented_options(self):
        smoke = ["/m/final/mojolearn-0.8.37-x.whl", "--expected-source-commit", COMMIT, "--out", "/m/out",
                 "--rent", "--vendor", "hip", "--provider", "auto", "--gpu", "NVIDIA L40S",
                 "--column", "/m/selection-hip.json",
                 "--ref-column", "/m/check/metal/column.json", "--ref-column", "/m/check/cpu/column.json",
                 "--plugin", "/m/final/mojolearn_amd-0.8.37-x.whl", "--plugin", "/m/final/mojolearn_nvidia-0.8.37-x.whl"]
        uploads, argv = lqs.box_plan(smoke, "/root/release-smoke/0.8.37/amd")
        rin = "/root/release-smoke/0.8.37/amd/in"
        self.assertEqual(argv, [
            "bash", "tools/release_wheel_smoke.sh", f"{rin}/mojolearn-0.8.37-x.whl",
            "--expected-source-commit", COMMIT, "--vendor", "hip", "--column", f"{rin}/selection-hip.json",
            "--ref-column", f"{rin}/ref-0-metal-column.json", "--ref-column", f"{rin}/ref-1-cpu-column.json",
            "--plugin", f"{rin}/mojolearn_amd-0.8.37-x.whl", "--plugin", f"{rin}/mojolearn_nvidia-0.8.37-x.whl",
            "--local", "--out", "/root/release-smoke/0.8.37/amd/out"])
        self.assertEqual([str(l) for l, _ in uploads][:2], ["/m/final/mojolearn-0.8.37-x.whl", "/m/selection-hip.json"])
        self.assertEqual(len(uploads), 6)

    def test_lq_line_queues_next_with_no_build(self):
        line = lqs.lq_line("/e/lq/lq", "nv", "main", "rel-0.8.37-nvidia-20261008T000000Z",
                           ["bash", "tools/release_wheel_smoke.sh", "w.whl", "--local"], 7200)
        self.assertEqual(line, ["/e/lq/lq", "add", "--front", "nv", "CMD", "main", "rel-0.8.37-nvidia-20261008T000000Z",
                                "bash", "tools/release_wheel_smoke.sh", "w.whl", "--local", "BUILDS=none",
                                "LQ_CMD_TIMEOUT=7200"])

    def test_result_line(self):
        text = ("n0042 nvidia main@abc CMD rel-0.8.37-nvidia-X rc=1 builds=[builds=none (skipped) ] digest=none last: FAILED\n"
                "n0043 nvidia main@abc CMD rel-0.8.37-nvidia-Y rc=0 builds=[] digest=none last: PASSED\n")
        self.assertEqual(lqs.result_line(text, "rel-0.8.37-nvidia-Y")[0], 0)
        self.assertEqual(lqs.result_line(text, "rel-0.8.37-nvidia-X")[0], 1)
        self.assertIsNone(lqs.result_line(text, "rel-0.8.37-nvidia-Z"))


class Run(unittest.TestCase):
    """The whole runner against a fake ssh/lq: upload, queue once, poll, fetch."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        (self.tmp / "lq").write_text(LQ_TEXT)
        self.wheel = self.tmp / "mojolearn-0.8.37-x.whl"
        self.wheel.write_bytes(b"core")
        self.sel = self.tmp / "selection-cuda.json"
        self.sel.write_text("{}")
        self.calls, self.polls = [], 0
        self.result_after = 2

    def tearDown(self):
        self._t.cleanup()

    def fake(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[0] == "ssh":
            remote = cmd[-1]
            if "echo READY" in remote:
                return SimpleNamespace(returncode=0, stdout="READY\n", stderr="")
            if remote.startswith("cat > "):
                kw["stdin"].read()
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            if remote.startswith("sha256sum "):
                rows = []
                for path in remote.split()[1:]:
                    local = self.wheel if path.endswith(".whl") else self.sel
                    rows.append(f"{hashlib.sha256(local.read_bytes()).hexdigest()}  {path}")
                return SimpleNamespace(returncode=0, stdout="\n".join(rows) + "\n", stderr="")
            if "tar czf" in remote:
                return SimpleNamespace(returncode=0, stdout=tgz({"results.json": b'{"status": "PASSED"}'}), stderr=b"")
        if cmd[0] == "tar":
            return subprocess.run(cmd, **kw)
        if cmd[1:3] == ["add", "--front"]:
            return SimpleNamespace(returncode=0, stdout="queued n0044 (gpu-queue 0044) FRONT\n", stderr="")
        if cmd[1] == "results":
            self.polls += 1
            tag = cmd[3]
            out = f"n0044 nvidia main@abc CMD {tag} rc=0 builds=[] digest=none last: PASSED\n" \
                if self.polls >= self.result_after else ""
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        raise AssertionError(f"unexpected command {cmd}")

    def runner(self, state):
        a = lqs.parse(["--box", "nv", "--tag", "rel-0.8.37-nvidia", "--remote-dir", "/root/release-smoke/0.8.37/nvidia",
                       "--out", str(self.tmp / "smoke-linux"), "--state", str(state), "--lq", str(self.tmp / "lq"),
                       "--poll-seconds", "1", "--", str(self.wheel), "--expected-source-commit", COMMIT,
                       "--out", "/ignored", "--rent", "--gpu", "NVIDIA L40S", "--column", str(self.sel)])
        return lqs.Runner(a, run=self.fake, sleep=lambda s: None)

    def test_upload_queue_poll_fetch(self):
        state = self.tmp / "columns" / "nvidia.lq.json"
        rc = self.runner(state).main()
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads((self.tmp / "smoke-linux" / "results.json").read_text())["status"], "PASSED")
        adds = [c for c in self.calls if c[1:3] == ["add", "--front"]]
        self.assertEqual(len(adds), 1)
        self.assertIn("--local", adds[0])
        self.assertNotIn("--rent", adds[0])
        self.assertIn("BUILDS=none", adds[0])
        ssh = [c for c in self.calls if c[0] == "ssh"]
        self.assertTrue(all(c[c.index("root@195.0.2.1") - 2:c.index("root@195.0.2.1")] == ["-p", "30401"] for c in ssh))
        self.assertFalse(state.exists(), "the state is dropped once the result is home")

    def test_a_relaunch_while_queued_polls_and_never_queues_again(self):
        state = self.tmp / "columns" / "nvidia.lq.json"
        r = self.runner(state)
        self.result_after = 10 ** 9
        r.a.max_wait_hours = 0          # gives up at once: the job stays queued
        self.assertEqual(r.main(), 3)
        tag = json.loads(state.read_text())["tag"]
        self.calls.clear()
        self.result_after = 0
        self.assertEqual(self.runner(state).main(), 0)
        self.assertFalse([c for c in self.calls if c[1:3] == ["add", "--front"]])
        self.assertTrue(any(c[1] == "results" and c[3] == tag for c in self.calls if c[0] != "ssh"))


def args(**kw):
    base = dict(version="0.8.14", dry_run=False, publish=None, only="", redo="", build_backend="gpu-legs",
                amd_expect_from="", smoke_gpu="", state_dir="", amd_build_provider=None, amd_provider="auto",
                cpu_column=False, smoke_via="lq", hopper_box="", smoke_branch="main")
    base.update(kw)
    return argparse.Namespace(**base)


class ReleaseRoute(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def release(self, **kw):
        r = release.Release(args(state_dir=str(self.tmp / "state"), **kw), runner=lambda *a, **k: 0)
        r.state["commit"] = COMMIT
        r.lines = []
        r.say = r.lines.append
        r.gpu_selection = lambda vendor: self.tmp / f"selection-{vendor}.json"
        return r

    def test_lq_is_the_default_and_the_hopper_column_does_not_gate_publish_nvidia(self):
        with mock.patch.object(release.Release, "go", lambda self: self.args):
            parsed = release.main(["0.8.14", "--state-dir", str(self.tmp / "s")])
            self.assertEqual((parsed.smoke_via, parsed.hopper_box, parsed.smoke_branch), ("lq", "", "main"))
            self.assertEqual(release.main(["0.8.14", "--state-dir", str(self.tmp / "s"), "--smoke-via", "rent"]).smoke_via,
                             "rent")
        r = self.release()
        self.assertNotIn("gpu-column-nvidia-hopper", r.NEEDS["publish-nvidia"])
        self.assertIn("gpu-column-nvidia-hopper", r.AFTER["publish-nvidia"])
        self.assertIn("SKIPPED: no Hopper box held", r.step_gpu_column_nvidia_hopper())
        self.assertEqual(r.smoke_notes("nvidia"), {"sm_90a": "not run (no Hopper box held)"})
        self.assertEqual(r.smoke_notes("amd"), {})

    def test_a_held_hopper_box_gates_publish_nvidia_again(self):
        r = self.release(hopper_box="-p 22 root@198.51.100.7")
        self.assertIn("gpu-column-nvidia-hopper", r.NEEDS["publish-nvidia"])
        self.assertEqual(r.smoke_notes("nvidia"), {})
        rent = self.release(smoke_via="rent")
        self.assertIn("gpu-column-nvidia-hopper", rent.NEEDS["publish-nvidia"])

    def test_lq_column_commands(self):
        r = self.release()
        cmd = r.lq_smoke_command("nvidia", "cuda", "/f/core.whl", self.tmp / "out", self.tmp / "columns",
                                 self.tmp / "selection-cuda.json", ["--ref-column", "/c/metal/column.json"])
        self.assertEqual(cmd[1:12], ["tools/release_lq_smoke.py", "--box", "nv", "--tag", "rel-0.8.14-nvidia",
                                     "--remote-dir", "/root/release-smoke/0.8.14/nvidia", "--out", str(self.tmp / "out"),
                                     "--state", str(self.tmp / "columns" / "nvidia.lq.json")])
        tail = cmd[cmd.index("--") + 1:]
        self.assertEqual(tail[:6], ["/f/core.whl", "--expected-source-commit", COMMIT, "--vendor", "cuda", "--column"])
        self.assertNotIn("--rent", cmd)
        amd = r.lq_smoke_command("amd", "hip", "/f/core.whl", self.tmp / "out", self.tmp / "columns",
                                 self.tmp / "selection-hip.json", [])
        self.assertEqual(amd[amd.index("--box") + 1], "amd")
        self.assertEqual(amd[amd.index("--vendor") + 1], "hip")


if __name__ == "__main__":
    unittest.main()
