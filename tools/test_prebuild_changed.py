#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""tools/prebuild_changed.py: the planner predicts the keys a release box
computes, and counts what the cache lacks.

The box is simulated with tools/bincache.py's own key_fields (the device arch
and OS injected, the toolchain read as linux-64 whatever runs the test) plus
the `declared` block cmd_build adds, so "predicted" is compared with the
shipped key code, not with a copy of it. The cache is a mocked lister; no R2,
no network, no Mojo, no GPU, no rental.

    python3 -m unittest -v tools/test_prebuild_changed.py
"""
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import bincache  # noqa: E402
import prebuild_changed as pc  # noqa: E402

IMAGE = "runpod:runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
OS = dict(cc="cc 11.4.0", glibc="glibc 2.35", id="ubuntu", ld="GNU ld 2.38", machine="x86_64", version_id="22.04")
LOCK = ("  - conda: https://conda.modular.com/max/linux-64/mojo-1.0.0-release.conda\n"
        "  - conda: https://conda.modular.com/max/osx-arm64/mojo-1.0.0-release.conda\n")


def real_tier_scripts():
    text = (ROOT / "packaging" / "linux" / "build_sets.sh").read_text()
    return re.search(r"^tier_scripts\(\) \{\n.*?^\}\n", text, re.M | re.S).group(0)


def build_script(name, src):
    return ("#!/bin/bash\nset -eu\n"
            "pixi run mojo build --emit shared-lib -I . bindings/%s -o python/mojolearn/x.so\n" % src)


def make_tree(root, gbdt_body="fn f():\n    pass\n"):
    """A small repository in the release layout: one tree binding (all tiers),
    one classical binding (fast + identical), one identical-only binding and
    one host family, each with a real `mojo build` line."""
    root = Path(root)
    (root / "bindings").mkdir(parents=True)
    (root / "packaging" / "linux").mkdir(parents=True)
    (root / "python" / "mojolearn").mkdir(parents=True)
    (root / "bindings" / "gbdt.mojo").write_text("from bindings.common import g\n" + gbdt_body)
    (root / "bindings" / "common.mojo").write_text("fn g():\n    pass\n")
    (root / "bindings" / "est.mojo").write_text("fn e():\n    pass\n")
    (root / "bindings" / "svm.mojo").write_text("fn s():\n    pass\n")
    (root / "bindings" / "blm.mojo").write_text("fn b():\n    pass\n")
    (root / "bindings" / "build_byte_lm.sh").write_text(build_script("byte_lm", "blm.mojo"))
    (root / "bindings" / "_mojolearn_alpha_host.mojo").write_text("fn a():\n    pass\n")
    (root / "bindings" / "build_gbdt.sh").write_text(build_script("gbdt", "gbdt.mojo"))
    (root / "bindings" / "build_estimators.sh").write_text(build_script("estimators", "est.mojo"))
    (root / "bindings" / "build_svm.sh").write_text(build_script("svm", "svm.mojo"))
    (root / "bindings" / "build_host_family.sh").write_text(
        "#!/bin/bash\nfam=$1\npixi run mojo build --emit shared-lib -I . -o out.so\n")
    (root / "bindings" / "build_alpha_host.sh").write_text(
        "#!/bin/bash\nexec bash \"$(dirname \"$0\")/build_host_family.sh\" alpha\n")
    (root / "pixi.toml").write_text("[workspace]\nname = \"x\"\n[tasks]\nprebuild = \"echo\"\n")
    (root / "pixi.lock").write_text(LOCK)
    (root / "python" / "mojolearn" / "host_surface.py").write_text(
        "import sys\nif '--wheel-families' in sys.argv:\n    print('alpha')\n")
    (root / "packaging" / "linux" / "build_sets.sh").write_text(
        '#!/usr/bin/env bash\n'
        'SCRIPTS="${MOJOLEARN_BUILD_SCRIPTS:-build_gbdt.sh}"\n'
        'FAST_CLASSICAL_SCRIPTS="build_estimators.sh"\n'
        'IDENTICAL_ONLY_SCRIPTS="build_svm.sh"\n'
        + real_tier_scripts())
    return root


def box_env(tier, script, arch="sm_90a"):
    """What tools/release061_remote_build.sh + build_sets.sh export (0.8.19)."""
    env = {"MODULAR_HOME": "/root/mojolearn/.pixi/envs/default/share/max", "MOJOLEARN_BUILD_PIXI_ENV": "default",
           "MOJOLEARN_COMPILE_JOBS": "2", "MOJOLEARN_LINUX_CPU": "x86-64-v3", "MOJOLEARN_NUMERIC_MODE": tier,
           "MOJOLEARN_PACKAGE_BYTE_LM": "1", "MOJOLEARN_SKIP_BUILD_GATE": "1", "MOJOLEARN_BUILD_JOBS": "4",
           "MOJOLEARN_BINCACHE_OUT": "/root/x"}
    if script.endswith("_host.sh"):
        env["MOJOLEARN_TARGET_COLUMN"] = "cpu"
    else:
        env.update(MOJOLEARN_TARGET_COLUMN="nvidia", MOJOLEARN_GPU_ARCHS=arch)
    return env


def box_key(tree, tier, script, arch="sm_90a", env_extra=None):
    """(key, fields) as tools/bincache.py cmd_build computes them on the box."""
    env = dict(box_env(tier, script, arch), **(env_extra or {}))
    path = str(Path(tree) / "bindings" / script)
    with mock.patch.object(bincache, "toolchain", lambda repo: pc.linux_toolchain(repo)):
        fields = bincache.key_fields(tree, path, [], env, IMAGE, dev_arch=arch, os_info=dict(OS))
    fields["declared"] = dict(inputs=bincache.local_inputs(tree, path),
                              outputs=[pc.declared_output(tier, script)], shell="bash")
    fields["repo_path"] = "/root/mojolearn"       # the box's checkout, whatever tree the test used
    return bincache.key_of(fields), fields


def write_keys(tree, keys_dir, builds, **kw):
    keys_dir = Path(keys_dir)
    keys_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for tier, script in builds:
        key, fields = box_key(tree, tier, script, **kw)
        (keys_dir / (key + ".json")).write_text(json.dumps(fields, indent=1, sort_keys=True))
        out.append(key)
    return out


def profiles_from(keys_dir):
    return {"cuda-sm_90a": dict(route="gemm-campaign7", served_by="test", templates=pc.capture(keys_dir))}


class Enumerate(unittest.TestCase):
    def test_small_tree_uses_the_real_tier_function(self):
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(td)
            got = pc.enumerate_builds(tree)
        self.assertEqual(got, [
            ("fast", "build_gbdt.sh"), ("fast", "build_estimators.sh"),
            ("deterministic", "build_gbdt.sh"),
            ("identical", "build_gbdt.sh"), ("identical", "build_svm.sh"), ("identical", "build_estimators.sh"),
            ("identical", "build_byte_lm.sh"), ("identical", "build_alpha_host.sh")])

    def test_this_checkout_enumerates_every_tier_and_the_host_families(self):
        got = pc.enumerate_builds(ROOT)
        tiers = {t for t, _ in got}
        self.assertEqual(tiers, {"fast", "deterministic", "identical"})
        self.assertTrue(any(s.endswith("_host.sh") for _, s in got))
        self.assertIn(("identical", "build_byte_lm.sh"), got)
        self.assertEqual(len(got), len(set(got)))

    def test_declared_output_matches_build_sets(self):
        self.assertEqual(pc.declared_output("fast", "build.sh"), "python/mojolearn/_mojolearn.so")
        self.assertEqual(pc.declared_output("identical", "build_svm.sh"),
                         "python/mojolearn/identical/_mojolearn_svm.so")
        self.assertEqual(pc.declared_output("identical", "build_gp_infer_host.sh"),
                         "python/mojolearn/host/_mojolearn_gp_infer_host.so")


class Predict(unittest.TestCase):
    def test_every_predicted_key_equals_the_box_key(self):
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(Path(td) / "t")
            builds = pc.enumerate_builds(tree)
            box = write_keys(tree, Path(td) / "keys", builds)
            facts = pc.CommitFacts(tree)
            tmpl = pc.capture(Path(td) / "keys")
            pred = [pc.predict(facts, tmpl[pc.kind_of(t, s)], t, s)[0] for t, s in builds]
        self.assertEqual(pred, box)

    def test_a_template_from_one_commit_predicts_the_next(self):
        """The profile carries no commit-side field: a source edit moves the
        predicted key exactly as it moves the box's."""
        with tempfile.TemporaryDirectory() as td:
            old = make_tree(Path(td) / "old")
            new = make_tree(Path(td) / "new", gbdt_body="fn f():\n    return\n")
            builds = [("fast", "build_gbdt.sh"), ("identical", "build_alpha_host.sh")]
            write_keys(old, Path(td) / "keys", builds)
            tmpl = pc.capture(Path(td) / "keys")
            box_new = [box_key(new, t, s)[0] for t, s in builds]
            pred_new = [pc.predict(pc.CommitFacts(new), tmpl[pc.kind_of(t, s)], t, s)[0] for t, s in builds]
            pred_old = [pc.predict(pc.CommitFacts(old), tmpl[pc.kind_of(t, s)], t, s)[0] for t, s in builds]
        self.assertEqual(pred_new, box_new)
        self.assertNotEqual(pred_new[0], pred_old[0])      # gbdt's closure moved
        self.assertEqual(pred_new[1], pred_old[1])         # the host family's did not

    def test_toolchain_is_linux_whatever_runs_the_planner(self):
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(td)
            tc = pc.linux_toolchain(tree)
        self.assertEqual(tc["platform"], "linux-64")
        self.assertEqual(tc["packages"], ["mojo-1.0.0-release"])


class Capture(unittest.TestCase):
    def test_refuses_two_builds_of_one_kind_with_different_box_fields(self):
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(Path(td) / "t")
            write_keys(tree, Path(td) / "keys", [("fast", "build_gbdt.sh")])
            write_keys(tree, Path(td) / "keys", [("fast", "build_estimators.sh")],
                       env_extra={"MOJOLEARN_NEW_FLAG": "1"})
            with self.assertRaises(SystemExit):
                pc.capture(Path(td) / "keys")

    def test_refuses_a_record_that_does_not_hash_to_its_name(self):
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(Path(td) / "t")
            (key,) = write_keys(tree, Path(td) / "keys", [("fast", "build_gbdt.sh")])
            p = Path(td) / "keys" / (key + ".json")
            f = json.loads(p.read_text())
            f["image"] = "other"
            p.write_text(json.dumps(f))
            with self.assertRaises(SystemExit):
                pc.capture(Path(td) / "keys")

    def test_partition_is_the_one_the_leg_stages(self):
        t = dict(device_arch="sm_89", image=IMAGE)
        self.assertEqual(pc.partition_of(t), "sm_89/runpod-runpod-pytorch-2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")


class CheckKeys(unittest.TestCase):
    def test_a_stale_profile_is_named_not_passed(self):
        """A leg that exported a new variable computes keys the planner does
        not: check-keys lists them as box-only (the rent path fails on it)."""
        with tempfile.TemporaryDirectory() as td:
            tree = make_tree(Path(td) / "t")
            builds = [("fast", "build_gbdt.sh"), ("identical", "build_alpha_host.sh")]
            write_keys(tree, Path(td) / "old", builds)
            tmpl = pc.capture(Path(td) / "old")
            write_keys(tree, Path(td) / "new", builds, env_extra={"MOJOLEARN_NEW_FLAG": "1"})
            ok, box_only, pred_only = pc.check_keys(pc.CommitFacts(tree), tmpl, Path(td) / "new", builds)
            ok2, box_only2, _ = pc.check_keys(pc.CommitFacts(tree), tmpl, Path(td) / "old", builds)
        self.assertEqual((len(ok), len(box_only), len(pred_only)), (0, 2, 2))
        self.assertEqual((len(ok2), box_only2), (2, []))


class Plan(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        d = Path(self.td.name)
        self.tree = make_tree(d / "t")
        self.builds = pc.enumerate_builds(self.tree)
        self.keys = write_keys(self.tree, d / "keys", self.builds)
        self.profiles = profiles_from(d / "keys")
        self.part = pc.partition_of(self.profiles["cuda-sm_90a"]["templates"]["host"])

    def lister(self, present):
        prefix = "%s/%s/" % (bincache.OBJECT_PREFIX, self.part)

        def ls(creds, pfx):
            self.assertEqual(pfx, prefix)
            return [(prefix + k + ".tar.gz", "2026-09-25T00:00:00Z", 1) for k in present] + \
                   [(prefix + "not-a-key.txt", "2026-09-25T00:00:00Z", 1)]
        return ls

    def test_counts_missing_and_cached(self):
        plan = pc.make_plan("c" * 40, self.tree, self.profiles, ["cuda-sm_90a"], {"R2_BUCKET": "b"},
                            lister=self.lister(self.keys[:3]))
        c = plan["targets"]["cuda-sm_90a"]["counts"]
        self.assertEqual((c["cached"], c["missing"]), (3, len(self.builds) - 3))
        self.assertIn("bindings to prebuild", pc.render(plan))
        self.assertIn("| cuda-sm_90a |", pc.markdown(plan))

    def test_nothing_to_do_when_every_key_is_cached(self):
        plan = pc.make_plan("c" * 40, self.tree, self.profiles, ["cuda-sm_90a"], {"R2_BUCKET": "b"},
                            lister=self.lister(self.keys))
        self.assertEqual(plan["targets"]["cuda-sm_90a"]["counts"]["missing"], 0)
        self.assertTrue(pc.render(plan).endswith("nothing to do"))

    def test_no_credentials_means_unknown_not_missing(self):
        plan = pc.make_plan("c" * 40, self.tree, self.profiles, ["cuda-sm_90a"], None)
        c = plan["targets"]["cuda-sm_90a"]["counts"]
        self.assertEqual((c["missing"], c["cached"], c["unknown"]), (0, 0, len(self.builds)))
        self.assertIn("unknown", pc.render(plan))

    def test_base_counts_only_the_bindings_whose_identity_moved(self):
        base = make_tree(Path(self.td.name) / "base", gbdt_body="fn f():\n    return\n")
        plan = pc.make_plan("c" * 40, self.tree, self.profiles, ["cuda-sm_90a"], None, base_tree=base)
        changed = sorted({r["script"] for r in plan["targets"]["cuda-sm_90a"]["rows"] if r["changed"]})
        self.assertEqual(changed, ["build_gbdt.sh"])


class Creds(unittest.TestCase):
    def test_env_then_file_then_none(self):
        names = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
        self.assertEqual(pc.r2_creds({n: "v" for n in names}), {n: "v" for n in names})
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "r2"
            f.write_text("".join("export %s='%s'\n" % (n, n.lower()) for n in names))
            self.assertEqual(pc.r2_creds({"MOJOLEARN_R2_CREDS": str(f)})["R2_BUCKET"], "r2_bucket")
            self.assertIsNone(pc.r2_creds({"MOJOLEARN_R2_CREDS": str(Path(td) / "absent")}))


class Wiring(unittest.TestCase):
    def test_committed_profiles_load_and_name_the_release_partitions(self):
        prof = pc.load_profiles()
        self.assertEqual(sorted(prof), ["cuda-sm_89", "cuda-sm_90a", "hip-gfx942"])
        for name, p in prof.items():
            self.assertEqual(sorted(p["templates"]), ["gpu:deterministic", "gpu:fast", "gpu:identical", "host"])
        self.assertTrue(pc.partition_of(prof["cuda-sm_89"]["templates"]["host"]).startswith("sm_89/runpod-"))
        self.assertTrue(pc.partition_of(prof["hip-gfx942"]["templates"]["host"]).startswith("none/runpod-cpu-"))

    def test_the_release_leg_promotes_what_build_sets_uploaded(self):
        text = (TOOLS / "gemm_remote_leg.sh").read_text()
        self.assertIn('"$OUT/remote/release-build/build/bincache"', text)
        self.assertRegex(text, r'NVIDIA_CAMPAIGN" = 7 \] && _bc_slots="\$\{MOJOLEARN_BINCACHE_SLOTS:-192\}"')

    def test_workflow_is_manual_only_and_never_rents_by_default(self):
        # Only light-checks starts by itself (light-checks.yml refuses any
        # other workflow with a push or schedule trigger).
        text = (ROOT / ".github" / "workflows" / "prebuild-bindings.yml").read_text()
        on = text.split("\non:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("workflow_dispatch:", on)
        for trigger in ("push:", "schedule:", "pull_request"):
            self.assertNotIn(trigger, on)
        self.assertIn("prebuild_changed.py plan", text)
        self.assertIn("if: ${{ inputs.rent }}", text)
        self.assertIn("PREBUILD_RENT_ENABLED", text)
        self.assertNotIn("--rent", text.split("- name: Plan", 1)[1].split("- name: Rent", 1)[0])


if __name__ == "__main__":
    unittest.main()
