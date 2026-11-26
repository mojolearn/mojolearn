"""Tooling tests for tools/group_reuse.py (lane build-reuse-1, 2026-11-26): the Mac predicts a merged group's
binding-cache key exactly as the build container computes it, every byte-deciding input moves it, two runners key
alike, the leg skips only groups whose predicted key is stored, and the leg's `group cache:` line. Pure Python over
temporary trees: nothing is compiled, dispatched or uploaded.

    python3 -m unittest -v tools/tests/test_group_reuse.py
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import bincache  # noqa: E402
import group_reuse as gr  # noqa: E402

OS = dict(id="ubuntu", version_id="22.04", glibc="glibc 2.35", machine="x86_64", ld="GNU ld 2.38", cc="cc 11.4")
IMAGE = "runpod-cpu:rocm/dev-ubuntu-22.04@sha256:" + "a" * 64
ARCH = "sm_89"
MERGED_BUILD = """#!/bin/bash
set -eu
# the route's flags, read from the environment
: "${MOJOLEARN_BUILD_DEFINES:-}${MOJOLEARN_BUILD_EXTRA_DEFINES:-}${MOJOLEARN_COMPILE_JOBS:-2}${MOJOLEARN_GPU_ARCHS:-}"
: "${MOJOLEARN_TARGET_COLUMN:-}"
echo pixi run mojo build -j 2 --emit shared-lib
"""
# the commit's own merged plan (bincache reads it through tools/merged_bindings.py of the tree)
MERGED_BINDINGS = '''
def plan(dir_name, plat, byte_lm, layout="release", repo=None):
    return [("_g_a", "default", ["_mojolearn_a"]), ("_g_b", "default", ["_mojolearn_b"])]


def root_text(stem, names, repo=None):
    return "root %s %s\\n" % (stem, ",".join(names))
'''
LOCK = "      - conda: https://conda.modular.com/max/linux-64/mojo-compiler-%s-release.conda\n"
#: what the route exports on every runner of a leg
ROUTE_ENV = {"MOJOLEARN_COMPILE_JOBS": "2", "MOJOLEARN_BUILD_PIXI_ENV": "default", "MODULAR_HOME": "/root/m/max"}
#: what differs between two runners of one leg and must not reach a key
PER_RUNNER = {"MOJOLEARN_BINCACHE_MAP": "/root/.mojolearn_bincache/urls.tsv", "MOJOLEARN_COMMIT": "c" * 40,
              "MOJOLEARN_BUILD_JOBS": "4", "MOJOLEARN_RELEASE_BUILD_SECONDS": "5123", "RUNNER_NAME": "GitHub Actions 7",
              "MOJOLEARN_BINCACHE_OUT": "/root/leg_out/bincache"}


def make_tree(root, compiler="1.2.0"):
    root = Path(root)
    files = {
        "bindings/merged_build.sh": MERGED_BUILD,
        "bindings/_mojolearn_a.mojo": "from core import x\n",
        "bindings/_mojolearn_b.mojo": "from core import y\n",
        "core/x.mojo": "fn x(): pass\n",
        "core/y.mojo": "fn y(): pass\n",
        "core/unrelated.mojo": "fn u(): pass\n",
        "tools/merged_bindings.py": MERGED_BINDINGS,
        "pixi.toml": "[workspace]\n",
        "pixi.lock": LOCK % compiler,
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root.resolve()


def args_of(group, kind="gpu"):
    return [kind, "--group", group, "--rootdir", "/tmp/mojolearn-merged-roots/identical"]


def box_fields(tree, group, env=None, gpu=ARCH, kind="gpu"):
    """The fields the build container computes (bincache build) for one group of TREE, through the commit's own
    merged plan, as the box does."""
    environ = dict(ROUTE_ENV, MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_GPU_ARCHS=gpu, MOJOLEARN_TARGET_COLUMN="nvidia")
    environ.update(env or {})
    out = "python/mojolearn/%s/%s.so" % ("host" if kind == "host" else "identical", group)
    with gr.commit_merged_bindings(tree):
        fields, _ = bincache.build_fields(tree, str(tree / "bindings/merged_build.sh"), args_of(group, kind), environ,
                                          IMAGE, declared=[out], shell="bash", dev_arch="none", os_info=dict(OS),
                                          plat="linux")
    fields["toolchain"] = gr.linux_toolchain(tree, "linux-64")
    return fields


def write_leg(root, version, fields_list, arch=ARCH):
    keys = Path(root) / version / "c12" / "legs" / ("cuda-%s" % arch) / "GHA" / "bincache" / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    for f in fields_list:
        (keys / (bincache.key_of(f) + ".json")).write_text(json.dumps(f))
    return keys


class Prediction(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.td, True)
        # the box's checkout and the Mac's source worktree of the same commit live at different paths
        self.box = make_tree(self.td / "box")
        self.mac = make_tree(self.td / "mac")

    def predicted(self, template, tree=None):
        got = gr.predict(tree or self.mac, template, gr.leg_environ({"x": template}))
        self.assertIsNotNone(got)
        return got[0]

    def test_an_unchanged_commit_predicts_the_recorded_key(self):
        rec = box_fields(self.box, "_g_a")
        self.assertEqual(self.predicted(rec), bincache.key_of(rec))

    def test_every_byte_deciding_input_moves_the_key_and_the_prediction_follows(self):
        rec = box_fields(self.box, "_g_a")
        old = bincache.key_of(rec)
        edits = {
            "closure": ("core/x.mojo", "fn x(): return\n"),
            "member root": ("bindings/_mojolearn_a.mojo", "from core import x\nfrom core import y\n"),
            "executed script line": ("bindings/merged_build.sh", MERGED_BUILD.replace("-j 2", "-j 3")),
            "toolchain": ("pixi.lock", LOCK % "1.3.0"),
        }
        for what, (rel, text) in edits.items():
            with self.subTest(what):
                # the box keeps its checkout at one path from commit to commit (repo_path is a key field, and the
                # prediction takes it from the leg's record), so the edited commit is checked out where the box was
                shutil.rmtree(self.td / "box")
                new_box, new_mac = make_tree(self.td / "box"), make_tree(self.td / ("mac-" + what))
                for t in (new_box, new_mac):
                    (t / rel).write_text(text)
                truth = bincache.key_of(box_fields(new_box, "_g_a"))
                self.assertNotEqual(truth, old)
                self.assertEqual(self.predicted(rec, new_mac), truth)

    def test_a_file_outside_the_closure_and_a_comment_keep_the_key(self):
        rec = box_fields(self.box, "_g_a")
        (self.mac / "core/unrelated.mojo").write_text("fn u(): return\n")
        (self.mac / "bindings/merged_build.sh").write_text(MERGED_BUILD.replace("# the route's", "# The route's"))
        (self.mac / "core/y.mojo").write_text("fn y(): return\n")        # group b's closure, not a's
        self.assertEqual(self.predicted(rec), bincache.key_of(rec))

    def test_flags_defines_and_target_are_the_records_and_each_moves_the_key(self):
        base = box_fields(self.box, "_g_a")
        variants = [box_fields(self.box, "_g_a", env={"MOJOLEARN_BUILD_EXTRA_DEFINES": "-D X=1"}),
                    box_fields(self.box, "_g_a", env={"MOJOLEARN_COMPILE_JOBS": "4"}),
                    box_fields(self.box, "_g_a", gpu="sm_80")]
        keys = {bincache.key_of(base)} | {bincache.key_of(v) for v in variants}
        self.assertEqual(len(keys), 4)
        for v in variants:
            self.assertEqual(self.predicted(v), bincache.key_of(v))

    def test_two_runners_of_one_leg_key_alike(self):
        one = box_fields(self.box, "_g_a")
        two = box_fields(self.box, "_g_a", env=PER_RUNNER)
        self.assertEqual(bincache.key_of(one), bincache.key_of(two))
        # and a record from either runner predicts the other's key
        self.assertEqual(self.predicted(two), bincache.key_of(one))

    def test_arm_and_fault_records_are_never_the_leg_half(self):
        good = box_fields(self.box, "_g_a")
        arm = box_fields(self.box, "_g_a", env={"MOJOLEARN_BUILD_DEFINES": "A=1"})
        fault = box_fields(self.box, "_g_a", env={"MOJOLEARN_SABOTAGE_SVD": "1"})
        self.assertTrue(gr.usable(good, ARCH))
        self.assertFalse(gr.usable(arm, ARCH))
        self.assertFalse(gr.usable(fault, ARCH))
        self.assertFalse(gr.usable(good, "sm_80"))                    # a device record of another target

    def test_a_group_the_commit_no_longer_plans_is_not_predicted(self):
        rec = box_fields(self.box, "_g_a")
        (self.mac / "tools/merged_bindings.py").write_text(MERGED_BINDINGS.replace('"_g_a"', '"_g_c"'))
        self.assertIsNone(gr.predict(self.mac, rec))


class SkipPlan(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.td, True)
        self.box = make_tree(self.td / "box")
        self.mac = make_tree(self.td / "mac")
        self.roots = [self.td / "work"]
        self.recs = [box_fields(self.box, "_g_a"), box_fields(self.box, "_g_b")]
        write_leg(self.roots[0], "0.8.67.dev0+aaaaaaaaaaaa", self.recs)
        self.entries = ["identical:merged_build.sh:_g_a", "identical:merged_build.sh:_g_b"]

    def test_only_stored_predicted_keys_lose_their_shard_job(self):
        stored = {bincache.key_of(self.recs[0])}
        plan = gr.skip_plan(self.mac, ARCH, self.entries, stored, self.roots)
        self.assertEqual(plan["skip"], ["identical:merged_build.sh:_g_a"])
        self.assertEqual(plan["keys"]["identical:merged_build.sh:_g_a"], bincache.key_of(self.recs[0]))
        self.assertIn("predicted_hits=1 shard_jobs=1", gr.plan_line(plan))

    def test_a_changed_closure_keeps_its_shard_job(self):
        (self.mac / "core/x.mojo").write_text("fn x(): return\n")
        stored = {bincache.key_of(r) for r in self.recs}
        plan = gr.skip_plan(self.mac, ARCH, self.entries, stored, self.roots)
        self.assertEqual(plan["skip"], ["identical:merged_build.sh:_g_b"])

    def test_no_record_or_another_arch_plans_no_skip(self):
        stored = {bincache.key_of(r) for r in self.recs}
        self.assertEqual(gr.skip_plan(self.mac, ARCH, self.entries, stored, [self.td / "empty"])["skip"], [])
        self.assertEqual(gr.skip_plan(self.mac, "sm_80", self.entries, stored, self.roots)["skip"], [])

    def test_a_record_that_does_not_hash_to_its_name_is_passed_over(self):
        keys = write_leg(self.roots[0], "0.8.67.dev0+bbbbbbbbbbbb", [])
        forged = dict(self.recs[0], gpu_archs=ARCH)
        (keys / ("f" * 64 + ".json")).write_text(json.dumps(forged))
        legs = gr.leg_records(self.roots, ARCH)
        self.assertEqual([len(r) for _, r in legs], [2])               # only the honest leg

    def test_the_map_carries_a_wanted_object_beyond_the_newest(self):
        part = "none/x"
        creds = dict(R2_ACCOUNT_ID="acct", R2_ACCESS_KEY_ID="id", R2_SECRET_ACCESS_KEY="secret", R2_BUCKET="b")
        listing = [("%s/%s/%s.tar.gz" % (bincache.OBJECT_PREFIX, part, ("%x" % i) * 64), "2026-11-%02dT00:00:00Z" % (i + 1), 1)
                   for i in range(3)]
        old = ("0" * 64)
        lines, n_get, _ = bincache.plan_lines(creds, part, IMAGE, "leg", 1, max_get=2, listing=listing,
                                              want_keys=[old])
        self.assertEqual(n_get, 3)
        self.assertTrue(any(l.startswith("get\t" + old) for l in lines))
        lines, n_get, _ = bincache.plan_lines(creds, part, IMAGE, "leg", 1, max_get=2, listing=listing)
        self.assertEqual(n_get, 2)


class CacheLine(unittest.TestCase):
    def test_counts_each_group_once(self):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, td, True)
        row = lambda outcome, key, s: "2026-11-26T00:00:00Z\tbindings/merged_build.sh\t%s\t%s\t%s\tx.so=1\n" % (
            outcome, key, s)
        shard = td / "s1.tsv"
        shard.write_text(row("miss+built-uploaded", "a" * 64, 300.4) + row("hit", "b" * 64, 0.5)
                         + row("miss+built-not-uploaded:no-slot", "c" * 64, 100))
        assemble = td / "as.tsv"
        assemble.write_text(row("hot-hit", "a" * 64, 0.1) + row("hot-hit", "b" * 64, 0.1)
                            + row("hit", "d" * 64, 0.4)                       # a group with no shard job
                            + row("miss+built-uploaded", "e" * 64, 50))       # a wrong prediction
        counts = gr.cache_counts([shard], [assemble])
        self.assertEqual(gr.cache_line(counts), "group cache: hits=2 misses=3 compiles=3 uploads=2 secs=450")


class Wiring(unittest.TestCase):
    """tools/release.py passes --predict-skips for a measurement build's A set only."""

    def ctx(self, version="0.8.67.dev0+0123456789ab", defines="", pack="github"):
        return SimpleNamespace(rel=Path("/w/v/c"), base=Path("/w/v"), commit="0" * 40, dry=False,
                               source_checkout=lambda create=True: Path("/w/v/source/c"),
                               args=SimpleNamespace(candidate_version=version, candidate_pack=pack,
                                                    candidate_defines=defines, candidate_ref="",
                                                    candidate_tooling_sha=""))

    def test_a_set_predicts_and_b_arm_and_release_do_not(self):
        import release
        legs = release.github_legs(self.ctx())
        self.assertTrue(legs and all("--predict-skips" in l.command for l in legs))
        cmd = legs[0].command
        self.assertEqual(cmd[cmd.index("--records-root") + 1], "/w")
        self.assertEqual(cmd[cmd.index("--source-tree") + 1], "/w/v/source/c")
        for ctx in (self.ctx(version=""), self.ctx(pack="here")):
            self.assertTrue(all("--predict-skips" not in l.command for l in release.github_legs(ctx)))
        # the host groups once per build: the other cuda leg takes the sm_89 leg's; the owner and hip legs do not
        by = {l.name: l.command for l in legs}
        self.assertEqual(by["cuda-sm_80"][by["cuda-sm_80"].index("--host-groups-from") + 1], "/w/v/c/legs/cuda-sm_89")
        self.assertTrue(all("--host-groups-from" not in c for n, c in by.items() if n != "cuda-sm_80"))

    def test_the_workflow_shares_host_groups_and_overlaps_its_setup(self):
        import yaml
        wf = yaml.safe_load((ROOT / ".github/workflows/release-linux-build.yml").read_text())
        self.assertIn("host_groups_run", wf[True]["workflow_dispatch"]["inputs"])
        for job in ("shard", "assemble"):
            names = [st.get("name", "") for st in wf["jobs"][job]["steps"]]
            self.assertIn("Free disk and pull the image (background)", names)
            self.assertLess(names.index("Wait for the disk cleanup"), names.index(next(n for n in names if n.startswith("Swap"))))
        names = [st.get("name", "") for st in wf["jobs"]["assemble"]["steps"]]
        self.assertLess(names.index("Host groups from the owner leg's run"), names.index("Assemble the set"))
        self.assertEqual(wf["jobs"]["assemble"]["permissions"], {"contents": "read", "actions": "read"})
        box = (ROOT / "tools/gha_release_box.sh").read_text()
        self.assertIn('[[ "$MODE" = shard ]] && export MOJOLEARN_RELEASE_ADAPTERS=0', box)
        self.assertIn("MOJOLEARN_RELEASE_", bincache.NON_BUILD_PREFIXES)     # so no group key moves

    def test_run_refuses_host_sharing_on_the_owner_or_without_prediction(self):
        import release_github_build as G
        with tempfile.TemporaryDirectory() as td:
            for extra in (["--arch", "sm_89", "--predict-skips", "--source-tree", td, "--host-groups-from", td],
                          ["--arch", "sm_80", "--host-groups-from", td]):
                with self.assertRaises(SystemExit):
                    G.main(["run", "--commit", "0" * 40, "--out", str(Path(td) / "o"), *extra])

    def test_owner_run_reads_the_owner_legs_run_of_the_same_commit(self):
        import release_github_build as G
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "GHA").mkdir()
            (Path(td) / "GHA" / "run-id.json").write_text(json.dumps(dict(databaseId=7, source_commit="a" * 40)))
            self.assertEqual(G.owner_run(td, "a" * 40, wait=0), 7)
            self.assertIsNone(G.owner_run(td, "b" * 40, wait=0, pause=0))

    def test_run_refuses_prediction_for_an_arm(self):
        import release_github_build as G
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                G.main(["run", "--commit", "0" * 40, "--arch", ARCH, "--out", str(Path(td) / "o"), "--defines", "A=1",
                        "--predict-skips", "--source-tree", td])


if __name__ == "__main__":
    unittest.main()
