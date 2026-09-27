"""The GitHub build route (--build-backend github, 2026-09-27): the workflow is
manual only and runs the CPU build box's image and overlay, the planner splits
each set exactly once, a job's URL map keeps only its own upload slots, a shard
builds only its own bindings, and `run` (with a stub gh) dispatches, waits,
downloads into the cpu-box leg layout and admits only a tree its proof names.
Stdlib only; no network, no GitHub, no Mojo.

    python3 -m pytest -q tools/test_release_github_build.py
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import release_github_build as rgb  # noqa: E402
import bincache  # noqa: E402

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("release", HERE / "release.py")
rel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rel)
C = "c" * 40
WF = ROOT / ".github/workflows/release-linux-build.yml"


class Workflow(unittest.TestCase):
    def test_manual_only_and_never_interpolates_the_map_url(self):
        text = WF.read_text()
        self.assertEqual(text.count("include-hidden-files: true"), 2)
        on = text.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertIn("workflow_dispatch:", on)
        for trigger in ("push:", "pull_request", "schedule:"):
            self.assertNotIn(trigger, on)
        self.assertNotIn("inputs.map_url", text)
        self.assertIn('--event "$GITHUB_EVENT_PATH"', text)

    def test_same_image_and_overlay_as_the_cpu_box(self):
        box = (ROOT / "tools/release_linux_build.sh").read_text()
        image = re.search(r"^IMAGE='([^']+)'", box, re.M).group(1)
        self.assertEqual(rgb.IMAGE, image)
        self.assertIn("IMAGE: " + image, WF.read_text())
        overlay = re.search(r'^OVERLAY="([^"]+)"', box, re.M).group(1).split()
        self.assertEqual(list(rgb.OVERLAY), overlay)
        gha_box = (ROOT / "tools/gha_release_box.sh").read_text()
        self.assertIn('OVERLAY="' + " ".join(overlay) + '"', gha_box)
        cpu_leg = (ROOT / "tools/runpod_cpu_leg.sh").read_text()
        self.assertIn("PIXI_VERSION=" + re.search(r"^PIXIVER=(\S+)", gha_box, re.M).group(1), cpu_leg)
        for ex in rgb.SOURCE_EXCLUDES:
            self.assertIn("'" + ex + "'", cpu_leg)
            self.assertIn(ex, WF.read_text())

    def test_partition_is_the_cpu_box_partition(self):
        # bincache_leg.sh's slug of "runpod-cpu:<image>", as 0.8.24's hip leg staged it
        self.assertEqual(rgb.partition(), "none/runpod-cpu-rocm-dev-ubuntu-22.04-sha256-"
                         "a3850e6638c6c390436ef1aacd72fd1359af36083ac823d5136818206998")


BUILD_SETS = r'''#!/usr/bin/env bash
TIERS="${MOJOLEARN_BUILD_TIERS:-fast deterministic identical}"
SCRIPTS="${MOJOLEARN_BUILD_SCRIPTS:-build_gbdt.sh build_rf.sh}"
EXT_NAMES="_mojolearn_gbdt _mojolearn_rf"
FAST_CLASSICAL_SCRIPTS="build.sh"
IDENTICAL_ONLY_SCRIPTS="build_mamba.sh"
PACKAGE_BYTE_LM=${MOJOLEARN_PACKAGE_BYTE_LM:-0}
tier_scripts() {
  printf '%s' "$SCRIPTS"
  if [[ "$1" = identical ]]; then printf ' %s' "$IDENTICAL_ONLY_SCRIPTS"; fi
  if [[ "$1" = identical || "$1" = fast ]]; then printf ' %s' "$FAST_CLASSICAL_SCRIPTS"; fi
  if [[ "$PACKAGE_BYTE_LM" = 1 && "$1" = identical ]]; then printf ' build_byte_lm.sh'; fi
  if [[ "$PACKAGE_BYTE_LM" = 1 && "$1" = identical ]]; then
    for f in $HOST_FAMILIES; do printf ' build_%s_host.sh' "$f"; done
  fi
  printf '\n'
}
HOST_FAMILIES="core svm"
host_so() { printf 'x'; }
echo SHOULD NOT RUN
'''


class Plan(unittest.TestCase):
    def test_the_sources_own_lists_split_once_each(self):
        src = Path(tempfile.mkdtemp())
        (src / "packaging/linux").mkdir(parents=True)
        (src / "packaging/linux/build_sets.sh").write_text(BUILD_SETS)
        builds = rgb.set_builds(src)
        names = sorted(e for e, _ in builds)
        self.assertEqual(names, sorted([
            "fast:build_gbdt.sh", "fast:build_rf.sh", "fast:build.sh",
            "deterministic:build_gbdt.sh", "deterministic:build_rf.sh",
            "identical:build_gbdt.sh", "identical:build_rf.sh", "identical:build_mamba.sh", "identical:build.sh",
            "identical:build_byte_lm.sh", "identical:build_core_host.sh", "identical:build_svm_host.sh"]))
        self.assertEqual({e for e, h in builds if h}, {"identical:build_core_host.sh", "identical:build_svm_host.sh"})
        shards = rgb.split(builds, 4)
        seen = [e for s in shards for e in s["only"].split()]
        self.assertEqual(sorted(seen), names)
        self.assertEqual(shards, rgb.split(list(reversed(builds)), 4))
        # the heaviest build goes first, onto shard 0
        heaviest = min(builds, key=lambda b: (-rgb.WEIGHTS.get(b[0], rgb.DEFAULT_HOST_WEIGHT if b[1] else rgb.DEFAULT_GPU_WEIGHT), b[0]))[0]
        self.assertIn(heaviest, shards[0]["only"].split())


class JobMap(unittest.TestCase):
    def test_only_this_jobs_slots_and_a_header_only_map_without_r2(self):
        d = Path(tempfile.mkdtemp())
        full = ["#partition\tp/q", "#image\ti", "#leg\tL", "get\t" + "a" * 64 + "\thttps://get"]
        full += ["put\t%03d\thttps://put/%03d" % (i, i) for i in range(3 * rgb.SLOTS_PER_JOB)]
        (d / "map.tsv").write_text("\n".join(full) + "\n")
        (d / "event.json").write_text(json.dumps({"inputs": {"map_url": (d / "map.tsv").as_uri()}}))
        rgb.main(["job-map", "--event", str(d / "event.json"), "--dest", str(d / "urls.tsv"),
                  "--set-index", "0", "--job", "1", "--jobs-per-set", "3"])
        m = bincache.read_map(d / "urls.tsv")
        self.assertEqual(len(m["get"]), 1)
        self.assertEqual([s for s, _ in m["put"]],
                         ["%03d" % i for i in range(rgb.SLOTS_PER_JOB, 2 * rgb.SLOTS_PER_JOB)])
        self.assertEqual(stat.S_IMODE(os.stat(d / "urls.tsv").st_mode), 0o600)
        (d / "event.json").write_text(json.dumps({"inputs": {"map_url": ""}}))
        rgb.main(["job-map", "--event", str(d / "event.json"), "--dest", str(d / "urls2.tsv"),
                  "--set-index", "0", "--job", "0", "--jobs-per-set", "3", "--leg", "99"])
        m = bincache.read_map(d / "urls2.tsv")
        self.assertEqual((m["get"], m["put"]), ({}, []))
        self.assertEqual(m["header"], {"partition": rgb.partition(), "image": rgb.IMAGE_DECL, "leg": "gha-99"})


class Shard(unittest.TestCase):
    def test_a_build_outside_the_shard_returns_without_running(self):
        d = Path(tempfile.mkdtemp())
        (d / "bindings").mkdir()
        marker = d / "ran"
        (d / "bindings/build_x.sh").write_text("echo ran > %s\n" % marker)
        env = dict(os.environ, MOJOLEARN_BINCACHE="0", MOJOLEARN_NUMERIC_MODE="fast",
                   MOJOLEARN_BINCACHE_SHARD_ONLY="identical:build_x.sh fast:build_y.sh")
        self.assertEqual(bincache.cmd_build([str(d / "bindings/build_x.sh")], env), 0)
        self.assertFalse(marker.exists())
        env["MOJOLEARN_NUMERIC_MODE"] = "identical"
        self.assertEqual(bincache.cmd_build([str(d / "bindings/build_x.sh")], env), 0)
        self.assertTrue(marker.exists())
        # the shard filter is not a key input
        self.assertNotIn("MOJOLEARN_BINCACHE_SHARD_ONLY", bincache.build_env(env))


class Backend(unittest.TestCase):
    def test_github_is_the_default_with_the_cpu_box_layout(self):
        self.assertIs(rel.BUILD_BACKENDS["github"], rel.github_legs)
        self.assertIn('ap.add_argument("--build-backend", default="github"', (HERE / "release.py").read_text())
        ctx = argparse.Namespace(rel=Path(tempfile.mkdtemp()), commit=C)
        legs = rel.github_legs(ctx)
        self.assertEqual([l.name for l in legs], ["cuda-sm_90a", "cuda-sm_89", "hip-gfx942"])
        for l in legs:
            self.assertEqual(l.command[:3], ["python3", "tools/release_github_build.py", "run"])
            out = Path(l.command[l.command.index("--out") + 1])
            self.assertEqual((l.release_build, l.out_dir), rel.leg_layout(ctx.rel / "legs", "cpu-box", l.name))
            self.assertEqual(out, l.out_dir)
        self.assertIn("github", rel.SELF_OVERLAID)


STUB = r'''#!/usr/bin/env python3
import json, os, shutil, sys
a = sys.argv[1:]
log = os.environ["STUB_LOG"]
open(log, "a").write(json.dumps(a) + "\n")
if a[:1] == ["api"]:
    print("7 .github/workflows/release-linux-build.yml")
elif a[:2] == ["workflow", "run"]:
    tag = [x for x in a if x.startswith("tag=")][0][4:]
    open(os.environ["STUB_TAG"], "w").write(tag)
elif a[:2] == ["run", "list"]:
    tag = open(os.environ["STUB_TAG"]).read()
    print(json.dumps([{"databaseId": 42, "displayTitle": "x [" + tag + "]", "headSha": os.environ["STUB_HEAD"], "url": "u"}]))
elif a[:2] == ["run", "view"]:
    print(json.dumps({"status": "completed", "conclusion": os.environ.get("STUB_CONCLUSION", "success"),
                      "url": "u", "jobs": [{"name": "assemble sm_89", "conclusion": os.environ.get("STUB_CONCLUSION", "success"),
                      "startedAt": "2026-09-27T00:00:00Z", "completedAt": "2026-09-27T00:10:00Z", "url": "j"}]}))
elif a[:2] == ["run", "download"]:
    if os.environ.get("STUB_NO_ARTIFACT"):
        sys.exit(1)
    shutil.copytree(os.environ["STUB_ART"], os.path.join(a[a.index("-D") + 1], "leg"))
else:
    sys.exit(3)
'''


class Run(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        art = self.d / "art"
        sets = art / "cuda-sm_89/release-build/build/sets/cuda/sm_89"
        (sets / "identical").mkdir(parents=True)
        (sets / "identical/_mojolearn_x.so").write_bytes(b"x")
        (sets / ".libs").mkdir()
        (sets / ".libs/libR.so").write_bytes(b"r")
        (sets / "manifest.json").write_text(json.dumps({"staged_libs": [
            {"name": "libR.so", "sha256": hashlib.sha256(b"r").hexdigest()}]}))
        proof = dict(complete=True, build_exit=0, source_commit=C,
                     extensions={"mojolearn/cuda/sm_89/identical/_mojolearn_x.so": hashlib.sha256(b"x").hexdigest()},
                     host_extension={})
        (art / "cuda-sm_89/release-build/build/build-provenance.json").write_text(json.dumps(proof))
        (art / "GHA").mkdir()
        gh = self.d / "gh"
        gh.write_text(STUB)
        gh.chmod(0o755)
        self.env = dict(STUB_LOG=str(self.d / "log"), STUB_TAG=str(self.d / "tag"), STUB_ART=str(art))
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self.gh, rgb.GH = rgb.GH, str(gh)
        self.tooling = rgb.tooling_ref
        rgb.tooling_ref = lambda ref: ("lane/x", "h" * 40)
        os.environ["STUB_HEAD"] = "h" * 40

    def tearDown(self):
        rgb.GH, rgb.tooling_ref = self.gh, self.tooling
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k in ("STUB_HEAD", "STUB_NO_ARTIFACT", "STUB_CONCLUSION"):
            os.environ.pop(k, None)

    def run_(self):
        return rgb.main(["run", "--commit", C, "--arch", "sm_89", "--out", str(self.d / "out"), "--no-bincache",
                         "--poll", "0"])

    def test_dispatch_wait_download_verify(self):
        self.assertEqual(self.run_(), 0)
        calls = [json.loads(l) for l in (self.d / "log").read_text().splitlines()]
        wr = next(c for c in calls if c[:2] == ["workflow", "run"])
        self.assertEqual(wr[2], "7")
        self.assertIn("commit=" + C, wr)
        self.assertIn("map_url=", wr)
        self.assertTrue((self.d / "out/cuda-sm_89/release-build/build/build-provenance.json").is_file())
        self.assertIn("TREE OK", (self.d / "out/GHA/summary.txt").read_text())
        self.assertIn("assemble sm_89\tsuccess", (self.d / "out/GHA/jobs.tsv").read_text())

    def test_a_tree_the_proof_does_not_name_is_refused(self):
        (Path(self.env["STUB_ART"]) / "cuda-sm_89/release-build/build/sets/cuda/sm_89/identical/_mojolearn_x.so").write_bytes(b"y")
        self.assertEqual(self.run_(), 1)
        self.assertIn("TREE REFUSED", (self.d / "out/GHA/summary.txt").read_text())

    def test_a_set_without_its_runtime_closure_is_refused(self):
        (Path(self.env["STUB_ART"]) / "cuda-sm_89/release-build/build/sets/cuda/sm_89/.libs/libR.so").unlink()
        self.assertEqual(self.run_(), 1)
        self.assertIn(".libs/libR.so", (self.d / "out/GHA/summary.txt").read_text())

    def test_no_artifact_names_the_failed_job(self):
        os.environ["STUB_NO_ARTIFACT"] = "1"
        os.environ["STUB_CONCLUSION"] = "failure"
        with self.assertRaisesRegex(SystemExit, "assemble sm_89 \\(failure\\)"):
            self.run_()

    def test_amd_refuses_to_build_cold(self):
        with self.assertRaisesRegex(SystemExit, "allow-cold-amd"):
            rgb.main(["run", "--commit", C, "--arch", "gfx942", "--out", str(self.d / "o2"), "--no-bincache"])


if __name__ == "__main__":
    unittest.main()
