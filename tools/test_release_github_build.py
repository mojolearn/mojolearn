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
import unittest.mock

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


def ptx_shipped():
    """Whether a release ships the NVIDIA PTX slot: gpu_plugins `ptx_shipped`
    (off since 0.8.39 while PyPI's 100 MiB mojolearn-nvidia limit stands)."""
    from verify_linux_surface_qualification import load_gpu_plugins
    return load_gpu_plugins().PLUGINS["cuda"].get("ptx_shipped", True)


#: The release's Linux legs in release_reuse.LINUX_SETS order, read from the registry.
RELEASE_LEGS = ["cuda-sm_89"] + (["cuda-sm_80"] if ptx_shipped() else []) + ["hip-gfx942"]


class Workflow(unittest.TestCase):
    def test_manual_only_and_never_interpolates_the_map_url(self):
        text = WF.read_text()
        # shard, assemble and the host producer each carry a .libs/ runtime closure
        self.assertEqual(text.count("include-hidden-files: true"), 3)
        on = text.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertIn("workflow_dispatch:", on)
        for trigger in ("push:", "pull_request", "schedule:"):
            self.assertNotIn(trigger, on)
        self.assertNotIn("inputs.map_url", text)
        self.assertNotIn("inputs.host_donor_url", text)
        self.assertIn('--event "$GITHUB_EVENT_PATH"', text)

    def test_merged_jobs_get_swap_modes_and_the_cache_only_without_r2(self):
        text = WF.read_text()
        self.assertEqual(text.count("gha_release_box.sh swap"), 3)   # shard, assemble and host
        self.assertIn("SWAP_GIB: ${{ matrix.swap_gib }}", text)
        self.assertEqual(text.count('GHA_RELEASE_MODES="$MODES"'), 2)
        self.assertEqual(text.count("steps.map.outputs.r2 == 'false'"), 3)  # restore, prune, save
        self.assertIn("timeout-minutes: 360", text)
        gha = (ROOT / "tools/gha_release_box.sh").read_text()
        self.assertIn('export MOJOLEARN_RELEASE_MODES="${GHA_RELEASE_MODES:-identical}"', gha)
        self.assertIn('export RELEASE_MERGED_TIMEOUT_SECONDS="$GHA_MERGED_TIMEOUT_SECONDS"', gha)
        sys.path.insert(0, str(ROOT / "tools"))
        import release_limits
        self.assertLessEqual(release_limits.LIMITS["GHA_MERGED_TIMEOUT_SECONDS"],
                             release_limits.LIMITS["CPU_BOX_RELEASE_BUILD_SECONDS"])
        # the release modes are never a cache-key field, so passing them keeps every key the cpu box's
        self.assertNotIn("MOJOLEARN_RELEASE_MODES", bincache.build_env({"MOJOLEARN_RELEASE_MODES": "identical"}))

    def test_the_host_job_runs_the_sources_own_producer_from_a_git_checkout(self):
        text = WF.read_text()
        host = text.split("\n  host:\n", 1)[1]
        self.assertIn('-v "$GITHUB_WORKSPACE/src:/root/mojolearn"', host)
        self.assertIn("gha_release_box.sh host host", host)
        self.assertIn("release_github_build.py donor --event", host)
        gha = (ROOT / "tools/gha_release_box.sh").read_text()
        self.assertIn('tools/release_host_build.py build --root "$R"', gha)
        self.assertIn("--pixi-env default", gha)
        self.assertIn("'patchelf==0.17.2.4'", gha)

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
TIERS="${MOJOLEARN_BUILD_TIERS:-fast identical}"
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


#: The same lists with build_sets.sh's merged switch and pool (MOJOLEARN_MERGED_BINDINGS, lane/merged-bindings).
MERGED_BUILD_SETS = BUILD_SETS.replace("host_so() { printf 'x'; }", """host_so() { printf 'x'; }
running=0
pool_jobs() {   # tier -> "tier:script" per build of that tier
  if [[ "$MERGED" = 1 ]]; then
    printf '%s:merged_build.sh\\n' "$1"
    if [[ "$PACKAGE_BYTE_LM" = 1 && "$1" = identical ]]; then printf 'host:merged_build.sh\\n'; fi
  else
    for s in $(tier_scripts "$1"); do printf '%s:%s\\n' "$1" "$s"; done
  fi
}
echo SHOULD NOT RUN EITHER
""").replace('PACKAGE_BYTE_LM=${MOJOLEARN_PACKAGE_BYTE_LM:-0}',
             'PACKAGE_BYTE_LM=${MOJOLEARN_PACKAGE_BYTE_LM:-0}\nMERGED=$(python3 packaging/linux/merged_layout.py enabled) || exit 2')


def merged_source(default="1"):
    src = Path(tempfile.mkdtemp())
    (src / "packaging/linux").mkdir(parents=True)
    (src / "packaging/linux/build_sets.sh").write_text(MERGED_BUILD_SETS)
    layout = (ROOT / "packaging/linux/merged_layout.py").read_text()
    if default != "1":
        layout = layout.replace('get(SWITCH, "1")', 'get(SWITCH, "%s")' % default)
    (src / "packaging/linux/merged_layout.py").write_text(layout)
    return src


class MergedPlan(unittest.TestCase):
    def test_the_merged_pool_is_planned_exactly(self):
        ps = rgb.plan_set(merged_source())
        self.assertTrue(ps["merged"])
        self.assertEqual(sorted(e for e, _ in ps["builds"]),
                         ["fast:merged_build.sh", "host:merged_build.sh", "identical:merged_build.sh"])
        self.assertEqual({e for e, h in ps["builds"] if h}, {"host:merged_build.sh"})
        # members are the per-binding builds each merged file replaces
        self.assertEqual(ps["members"], {"fast:merged_build.sh": 3, "identical:merged_build.sh": 5,
                                         "host:merged_build.sh": 2})
        w = lambda e: rgb.WEIGHTS.get(e, rgb.DEFAULT_GPU_WEIGHT)
        self.assertEqual(ps["weights"]["fast:merged_build.sh"],
                         w("fast:build_gbdt.sh") + w("fast:build_rf.sh") + w("fast:build.sh"))
        self.assertEqual(ps["weights"]["host:merged_build.sh"],
                         rgb.WEIGHTS["identical:build_core_host.sh"] + rgb.WEIGHTS["identical:build_svm_host.sh"])
        self.assertEqual(len(ps["per_binding"]), 10)
        # the planner ignores the caller's switch: the box gets none
        old = os.environ.get("MOJOLEARN_MERGED_BINDINGS")
        os.environ["MOJOLEARN_MERGED_BINDINGS"] = "0"
        try:
            self.assertTrue(rgb.plan_set(merged_source())["merged"])
        finally:
            os.environ.pop("MOJOLEARN_MERGED_BINDINGS") if old is None else os.environ.__setitem__(
                "MOJOLEARN_MERGED_BINDINGS", old)

    def test_a_per_binding_source_plans_its_scripts_through_pool_jobs(self):
        ps = rgb.plan_set(merged_source(default="0"))
        self.assertFalse(ps["merged"])
        self.assertEqual(sorted(e for e, _ in ps["builds"]), sorted(e for e, _ in ps["per_binding"]))
        self.assertEqual(len(ps["builds"]), 10)

    def run_plan(self, src, *extra):
        d = Path(tempfile.mkdtemp())
        with unittest.mock.patch.object(rgb, "check_counts", lambda src, builds, modes=None: None):
            rc = rgb.main(["plan", "--src", str(src), "--sets", "sm_89 gfx942", "--shards", "6",
                           "--json", str(d / "plan.json"), *extra])
        return rc, json.loads((d / "plan.json").read_text())

    def test_one_merged_compile_per_shard_with_swap_for_its_peak(self):
        rc, doc = self.run_plan(merged_source())
        self.assertEqual(rc, 0)
        self.assertEqual((doc["layout"], doc["shards"], doc["builds"]), ("merged", 3, 3))
        self.assertGreaterEqual(doc["jobs_per_set"], doc["shards"] + 1)
        rows = doc["matrix"]["include"]
        self.assertEqual(len(rows), 6)
        for arch, vendor in (("sm_89", "cuda"), ("gfx942", "hip")):
            mine = [r for r in rows if r["arch"] == arch]
            self.assertTrue(all(r["n"] == 1 for r in mine))
            self.assertEqual(sorted(r["only"] for r in mine),
                             ["fast:merged_build.sh", "host:merged_build.sh", "identical:merged_build.sh"])
            for r in mine:
                tier = r["only"].split(":")[0]
                self.assertEqual(r["peak_gib"], rgb.UNIT_PEAK_GIB[(tier, vendor)])
                self.assertEqual(r["swap_gib"], rgb.swap_gib(r["peak_gib"]))
                self.assertEqual(r["unit"], tier + "-merged_build")
            # both merged compiles page past the runner's RAM (the host one was killed at 16 GB
            # without swap, run 40272710760)
            for r in mine:
                if r["only"] != "fast:merged_build.sh":
                    self.assertGreater(r["swap_gib"] + rgb.RUNNER_RAM_GIB, r["peak_gib"])
        self.assertEqual(doc["assemble_swap_gib"], max(r["swap_gib"] for r in rows))
        # the heaviest (identical) goes onto shard 0
        self.assertEqual(rows[0]["only"], "identical:merged_build.sh")

    def test_the_release_modes_choose_the_tiers(self):
        rc, doc = self.run_plan(merged_source(), "--modes", "identical")
        self.assertEqual(rc, 0)
        self.assertEqual(doc["modes"], ["identical"])
        self.assertEqual(sorted(r["only"] for r in doc["matrix"]["include"] if r["arch"] == "sm_89"),
                         ["host:merged_build.sh", "identical:merged_build.sh"])

    def test_a_compile_no_free_runner_can_hold_is_refused_with_one_sentence(self):
        big = dict(rgb.UNIT_PEAK_GIB, **{("identical", "cuda"): 100})
        with unittest.mock.patch.object(rgb, "UNIT_PEAK_GIB", big), \
                unittest.mock.patch.object(rgb, "check_counts", lambda src, builds, modes=None: None):
            with self.assertRaises(SystemExit) as cm:
                rgb.main(["plan", "--src", str(merged_source()), "--sets", "sm_89", "--shards", "6"])
        msg = str(cm.exception)
        self.assertIn("identical:merged_build.sh", msg)
        self.assertIn("tools/release_linux_build.sh", msg)
        self.assertEqual(msg.count(". "), 0, "one sentence")

    def test_the_planning_peaks_cover_every_recorded_peak(self):
        # 0.8.57 cpu box (guard): sm_89 48.59, sm_80 48.12, gfx942 47.87 GiB; DO MI325X 2026-10-20:
        # fast 36.6 GB; the host compile was killed at 16.0 GB on a free runner (run 40272710760)
        for key, peak in ((("identical", "cuda"), 48.59), (("identical", "hip"), 47.87),
                          (("fast", "cuda"), 36.6), (("fast", "hip"), 36.6), (("host", "cuda"), 16.0),
                          (("host", "hip"), 16.0)):
            self.assertGreaterEqual(rgb.UNIT_PEAK_GIB[key], peak)
            self.assertLessEqual(rgb.swap_gib(rgb.UNIT_PEAK_GIB[key]), rgb.SWAP_MAX_GIB)

    def test_the_host_producer_needs_a_donor(self):
        with self.assertRaisesRegex(SystemExit, "donor"):
            rgb.build_plan(merged_source(), ["host"], 6)
        doc = rgb.build_plan(merged_source(), ["host"], 6, host_donor=True)
        self.assertEqual((doc["sets"], doc["host"], doc["matrix"]["include"]), ([], True, []))
        self.assertEqual(doc["host_swap_gib"], rgb.swap_gib(rgb.HOST_PRODUCER_PEAK_GIB))
        self.assertGreater(doc["host_swap_gib"], 0)
        with unittest.mock.patch.object(rgb, "check_counts", lambda src, builds, modes=None: None):
            doc = rgb.build_plan(merged_source(), ["sm_89", "host"], 6)
        self.assertEqual((doc["sets"], doc["host"]), (["sm_89"], True))
        self.assertIn("host", doc["estimate"]["per_set"])

    def test_the_real_source_plans_per_its_layout(self):
        merged = rgb.source_merged(ROOT)
        if not merged:
            self.skipTest("this checkout builds per-binding")
        ps = rgb.plan_set(ROOT, ("identical",))
        self.assertEqual(sorted(e for e, _ in ps["builds"]), ["host:merged_build.sh", "identical:merged_build.sh"])
        rgb.check_counts(ROOT, ps["per_binding"], ("identical",))
        doc = rgb.build_plan(ROOT, ["sm_89", "sm_80", "gfx942"], 6, ("identical",))
        self.assertEqual(len(doc["matrix"]["include"]), 6)
        self.assertEqual(doc["estimate"]["total"]["jobs"], 1 + 3 * 3)

    def test_layout_reads_the_commit_without_a_checkout(self):
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.assertEqual(rgb.commit_merged(head), rgb.source_merged(ROOT))
        self.assertFalse(rgb.commit_merged("0" * 40))   # no such tree: per-binding
        self.assertFalse(rgb.commit_merged(head, {"MOJOLEARN_MERGED_BINDINGS": "0"}))


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
        # the job learns whether R2 is its cache (else actions/cache)
        rgb.main(["job-map", "--event", str(d / "event.json"), "--dest", str(d / "urls3.tsv"),
                  "--set-index", "0", "--job", "0", "--jobs-per-set", "3", "--github-output", str(d / "out")])
        self.assertEqual((d / "out").read_text(), "r2=false\n")

    def test_the_donor_is_verified_by_its_sha256(self):
        import tarfile
        d = Path(tempfile.mkdtemp())
        (d / "src/build").mkdir(parents=True)
        (d / "src/build/build-provenance.json").write_text(json.dumps({"source_commit": C}))
        with tarfile.open(d / "donor.tar.gz", "w:gz") as tf:
            tf.add(str(d / "src/build"), arcname="build")
        good = hashlib.sha256((d / "donor.tar.gz").read_bytes()).hexdigest()
        for sha, ok in (("0" * 64, False), (good, True)):
            (d / "event.json").write_text(json.dumps({"inputs": {"host_donor_url": (d / "donor.tar.gz").as_uri(),
                                                                 "host_donor_sha256": sha}}))
            dest = d / ("ok" if ok else "bad")
            if ok:
                self.assertEqual(rgb.main(["donor", "--event", str(d / "event.json"), "--dest", str(dest),
                                           "--github-output", str(d / "gho")]), 0)
                self.assertEqual((d / "gho").read_text(), "revision=%s\n" % C)
                self.assertTrue((dest / "build/build-provenance.json").is_file())
            else:
                with self.assertRaisesRegex(SystemExit, "the dispatch named"):
                    rgb.main(["donor", "--event", str(d / "event.json"), "--dest", str(dest)])
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

    def test_the_merged_host_build_is_its_own_shard_entry(self):
        # bindings/merged_build.sh host runs under MOJOLEARN_NUMERIC_MODE=identical, like the identical tier
        self.assertEqual(bincache.shard_entry("identical", "bindings/merged_build.sh", ["host"]), "host:merged_build.sh")
        self.assertEqual(bincache.shard_entry("identical", "bindings/merged_build.sh", ["gpu"]),
                         "identical:merged_build.sh")
        self.assertEqual(bincache.shard_entry("fast", "bindings/build_x.sh", []), "fast:build_x.sh")
        d = Path(tempfile.mkdtemp())
        (d / "bindings").mkdir()
        marker = d / "ran"
        (d / "bindings/merged_build.sh").write_text('echo "$1" >> %s\n' % marker)
        env = dict(os.environ, MOJOLEARN_BINCACHE="0", MOJOLEARN_NUMERIC_MODE="identical",
                   MOJOLEARN_BINCACHE_SHARD_ONLY="host:merged_build.sh")
        self.assertEqual(bincache.cmd_build([str(d / "bindings/merged_build.sh"), "gpu"], env), 0)
        self.assertFalse(marker.exists())
        self.assertEqual(bincache.cmd_build([str(d / "bindings/merged_build.sh"), "host"], env), 0)
        self.assertEqual(marker.read_text().split(), ["host"])
        # the shard judge on the runner names it the same way
        gha = (ROOT / "tools/gha_release_box.sh").read_text()
        self.assertIn('if name == "merged_build.sh" and (doc.get("args") or [])[:1] == ["host"]:', gha)


class Backend(unittest.TestCase):
    def test_github_legs_use_the_cpu_box_layout(self):
        # the default follows the source's layout: github for per-binding (test_release_cpu_box.py)
        self.assertIs(rel.BUILD_BACKENDS["github"], rel.github_legs)
        self.assertEqual(rel.default_build_backend(C), "github")
        ctx = argparse.Namespace(rel=Path(tempfile.mkdtemp()), commit=C)
        legs = rel.github_legs(ctx)
        # sm_80 is the PTX slot, built like any arch (Andrew 2026-10-10), while the release ships it
        self.assertEqual([l.name for l in legs], RELEASE_LEGS)
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
    if a[a.index("-n") + 1] == "host-build":
        shutil.copytree(os.environ["STUB_HOST_ART"], a[a.index("-D") + 1], dirs_exist_ok=True)
    else:
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
        self.assertIn("sets=sm_89", wr)
        self.assertIn("modes=identical", wr)
        self.assertIn("host_donor_url=", wr)
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

    def test_the_release_modes_reach_the_dispatch(self):
        os.environ["MOJOLEARN_RELEASE_MODES"] = "fast identical"
        try:
            self.assertEqual(self.run_(), 0)
        finally:
            os.environ.pop("MOJOLEARN_RELEASE_MODES")
        calls = [json.loads(l) for l in (self.d / "log").read_text().splitlines()]
        self.assertIn("modes=fast identical", next(c for c in calls if c[:2] == ["workflow", "run"]))

    def host_art(self, commit=C, data=b"h"):
        h = self.d / "host-art"
        (h / "set/host").mkdir(parents=True, exist_ok=True)
        (h / "set/host/_mojolearn_native_host.so").write_bytes(b"h")
        proof = dict(schema="mojolearn.linux.host-build.v1", complete=True, build_exit=0, source_commit=commit,
                     host_extension={"mojolearn/cpu/host/_mojolearn_native_host.so": hashlib.sha256(data).hexdigest()})
        (h / "host-build-proof.json").write_text(json.dumps(proof))
        os.environ["STUB_HOST_ART"] = str(h)

    def test_the_host_producer_is_downloaded_and_verified(self):
        self.host_art()
        try:
            rc = rgb.main(["run", "--commit", C, "--arch", "sm_89", "--host", "--out", str(self.d / "out"),
                           "--no-bincache", "--poll", "0"])
        finally:
            os.environ.pop("STUB_HOST_ART")
        self.assertEqual(rc, 0)
        calls = [json.loads(l) for l in (self.d / "log").read_text().splitlines()]
        self.assertIn("sets=sm_89 host", next(c for c in calls if c[:2] == ["workflow", "run"]))
        summary = (self.d / "out/GHA/summary.txt").read_text()
        self.assertIn("HOST OK", summary)
        self.assertIn("retained_release_composition.py", summary)

    def test_a_host_the_proof_does_not_name_is_refused(self):
        self.host_art(data=b"other")
        try:
            rc = rgb.main(["run", "--commit", C, "--arch", "sm_89", "--host", "--out", str(self.d / "out"),
                           "--no-bincache", "--poll", "0"])
        finally:
            os.environ.pop("STUB_HOST_ART")
        self.assertEqual(rc, 1)
        self.assertIn("HOST REFUSED", (self.d / "out/GHA/summary.txt").read_text())

    def test_host_only_needs_a_donor(self):
        with self.assertRaisesRegex(SystemExit, "host-donor"):
            rgb.main(["run", "--commit", C, "--host", "--out", str(self.d / "o3"), "--no-bincache"])

    def test_amd_refuses_to_build_cold(self):
        with self.assertRaisesRegex(SystemExit, "allow-cold-amd"):
            rgb.main(["run", "--commit", C, "--arch", "gfx942", "--out", str(self.d / "o2"), "--no-bincache"])


if __name__ == "__main__":
    unittest.main()
