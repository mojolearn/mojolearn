"""THE SPLIT LINUX PACKAGES IN `pixi run release` (--split-linux), proved
without a rental: the layout switch (OFF by default), the per-package
pipelines and their gates (each plugin on its own vendor's column, the core
LAST, after both plugins published, so only when both columns passed; all
three held by a divergent joint diff), the pack of the
three wheels through audit.sh and the strip, the columns' --plugin, and the
per-package publication. The runner stands in for every process; the pack
stand-in is the real packer (profile split) over test_split_wheels.py's
inert fake sets, so split_audit reads real split wheels."""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import unittest
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packaging" / "linux"))
import release  # noqa: E402
import test_release_severable as sev  # noqa: E402

HEAD = sev.HEAD


class SplitBase(sev.Base):
    def release(self, commit=sev.Y, split=True, **kw):
        r = super().release(commit=commit, split_linux=split, **kw)
        return r


class Switch(SplitBase):
    def test_off_by_default_and_the_combined_layout_is_unchanged(self):
        with mock.patch.dict(os.environ, {release.SPLIT_LINUX_ENV: ""}):
            r = sev.Base.release(self)
        self.assertFalse(r.split)
        self.assertEqual(r.STEPS, release.Release.STEPS)
        self.assertIn("publish-linux", r.STEPS)
        self.assertNotIn("publish-nvidia", r.STEPS)

    def test_the_environment_turns_it_on_and_the_flags_override(self):
        with mock.patch.dict(os.environ, {release.SPLIT_LINUX_ENV: "1"}):
            self.assertTrue(sev.Base.release(self).split)
            self.assertFalse(self.release(split=False).split)
        self.assertTrue(self.release().split)
        self.assertEqual(release.main.__code__.co_varnames[0], "argv")
        with mock.patch.object(release.Release, "go", lambda self: int(self.split)):
            with mock.patch.dict(os.environ, {release.SPLIT_LINUX_ENV: ""}):
                self.assertEqual(release.main(["0.8.99", "--dry-run", "--state-dir", str(self.tmp / "s1")]), 0)
                self.assertEqual(release.main(["0.8.99", "--dry-run", "--split-linux",
                                               "--state-dir", str(self.tmp / "s2")]), 1)
            with mock.patch.dict(os.environ, {release.SPLIT_LINUX_ENV: "1"}):
                self.assertEqual(release.main(["0.8.99", "--dry-run", "--combined-linux",
                                               "--state-dir", str(self.tmp / "s3")]), 0)

    def test_a_packed_layout_is_pinned(self):
        r = self.release()
        r.state["linux_layout"] = "split"
        r.save()
        with self.assertRaisesRegex(SystemExit, "--split-linux"):
            self.release(split=False)
        self.assertFalse(self.release(split=False, redo="linux-pack").split)

    def test_the_split_pipelines(self):
        r = self.release()
        self.assertEqual(list(r.PIPELINES), ["macos", "core-linux", "nvidia", "amd"])
        for p in r.PIPELINES.values():
            for s in p["builds"] + p["checks"] + [p["publish"]]:
                self.assertIn(s, r.STEPS)
        # THE PLUGINS FIRST, THE CORE LAST: the core requires both plugins
        self.assertEqual(set(r.NEEDS["publish-nvidia"]), {"gpu-column-nvidia", "linux-joint-diff"})
        self.assertEqual(set(r.NEEDS["publish-amd"]), {"gpu-column-amd", "linux-joint-diff"})
        self.assertEqual(set(r.NEEDS["publish-core-linux"]), {"linux-joint-diff", "publish-nvidia", "publish-amd"})
        self.assertEqual(set(r.AFTER["linux-joint-diff"]), {"gpu-column-nvidia", "gpu-column-amd"})


class Gates(SplitBase):
    """Every step stood in for; which packages publish when a column, the
    joint diff or the core's publish fails."""

    def staged(self, fail=()):
        r = self.release(publish="pypi")
        order, lock = [], threading.Lock()

        def make(step):
            def fn():
                with lock:
                    order.append(step)
                if step in fail:
                    raise release.StepFailed("boom " + step)
                if step.startswith("publish-"):
                    return "pypi via alpha-api-tag"
                if step in ("finish-line", "record"):
                    got = r.published_platforms()
                    return f"{step} for {','.join(got)}", dict(platforms=got)
                return "ok"
            return fn
        for step in r.STEPS:
            setattr(r, "step_" + step.replace("-", "_"), make(step))
        return r, order

    def published(self, r):
        return sorted(r.published_platforms())

    def test_everything_passing_publishes_the_three_packages_and_macos(self):
        r, order = self.staged()
        self.assertEqual(r.go(), 0)
        self.assertEqual(self.published(r), ["amd", "linux", "macos", "nvidia"])
        # the plugins first, the core last: pip resolves mojolearn==<v> only
        # once both plugins at <v> are on the index
        self.assertGreater(order.index("publish-core-linux"), order.index("publish-nvidia"))
        self.assertGreater(order.index("publish-core-linux"), order.index("publish-amd"))

    def test_a_failed_amd_column_holds_amd_and_the_core(self):
        r, order = self.staged(fail={"gpu-column-amd"})
        self.assertEqual(r.go(), 1)
        # the NVIDIA plugin may upload (unresolvable alone, it requires the core)
        self.assertEqual(self.published(r), ["macos", "nvidia"])
        self.assertNotIn("publish-core-linux", order)
        self.assertIn("amd: FAILED at gpu-column-amd", "\n".join(r.lines))

    def test_a_failed_nvidia_column_holds_nvidia_and_the_core(self):
        r, order = self.staged(fail={"gpu-column-nvidia"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "macos"])
        self.assertNotIn("publish-core-linux", order)

    def test_a_divergent_joint_diff_holds_all_three(self):
        r, order = self.staged(fail={"linux-joint-diff"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["macos"])
        for s in ("publish-core-linux", "publish-nvidia", "publish-amd"):
            self.assertNotIn(s, order)

    def test_a_failed_nvidia_publish_holds_the_core(self):
        r, order = self.staged(fail={"publish-nvidia"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "macos"])
        self.assertNotIn("publish-core-linux", order)

    def test_a_failed_amd_publish_holds_the_core(self):
        r, order = self.staged(fail={"publish-amd"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["macos", "nvidia"])
        self.assertNotIn("publish-core-linux", order)

    def test_a_failed_core_publish_leaves_both_plugins_published(self):
        r, order = self.staged(fail={"publish-core-linux"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "macos", "nvidia"])

    def test_the_joint_diff_waits_for_both_columns_to_settle(self):
        r, order = self.staged(fail={"gpu-column-nvidia"})
        r.go()
        self.assertGreater(order.index("linux-joint-diff"), order.index("gpu-column-amd"))
        self.assertGreater(order.index("linux-joint-diff"), order.index("gpu-column-nvidia"))


class Pack(SplitBase):
    """linux-pack --split-linux: pack (the real packer over fake sets), audit
    each wheel core first, strip each, split_audit the final set."""

    def test_three_wheels_packed_audited_stripped_and_audited_as_a_set(self):
        import test_split_wheels as tsw
        sets_root = self.tmp / "fake"
        set_dirs = tsw.make_sets(sets_root)
        calls = []

        def runner(cmd, env, log, detach=False):
            cmd = [str(c) for c in cmd]
            calls.append(cmd)
            if "pack-linux-wheel" in cmd:
                out = cmd[cmd.index("--out") + 1]
                args = [a for s in set_dirs for a in ("--set", s)]
                return tsw.pw.main(args + ["--out", out], _gates=False)
            if cmd[:2] == ["bash", "packaging/linux/audit.sh"]:
                whl = pathlib.Path(cmd[2])
                (whl.parent / "audit" / "repaired").mkdir(parents=True, exist_ok=True)
                shutil.copy2(whl, whl.parent / "audit" / "repaired" / whl.name)
                return 0
            if cmd[1].endswith("strip_wheel_dir_entries.py"):
                shutil.copy2(cmd[2], cmd[3])
                return 0
            return 0
        r = self.release(commit=HEAD)
        r.runner = runner
        r.pack_inputs = lambda: ([], [], ["m.json"])
        r.assembly_needed = lambda: False
        r.source_checkout = lambda create=True: ROOT
        result = r.step_linux_pack()
        finals = r.split_finals()
        self.assertTrue(all(finals.values()), finals)
        self.assertIn("--profile", calls[0])
        self.assertEqual(calls[0][calls[0].index("--profile") + 1], "release-split")
        audits = [pathlib.Path(c[2]).name.split("-")[0] for c in calls if c[:2] == ["bash", "packaging/linux/audit.sh"]]
        self.assertEqual(audits, ["mojolearn", "mojolearn_nvidia", "mojolearn_amd"], "the core first")
        self.assertEqual(sum(1 for c in calls if c[1].endswith("strip_wheel_dir_entries.py")), 3)
        report = json.loads((r.rel / "linux" / "final" / "split-audit.json").read_text())
        self.assertEqual(report["problems"], [])
        self.assertEqual(r.state["linux_layout"], "split")
        self.assertIn("mojolearn_amd-", result)
        # done means done: a rerun takes the three finals as they are
        calls.clear()
        self.assertTrue(r.step_linux_pack().startswith("have "))
        self.assertEqual(calls, [])


def fake_wheel(path, commit=None, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        if commit:
            z.writestr("mojolearn/identity_columns/COMMIT", commit + "\n")
        z.writestr("payload", data)
    return path


class ColumnsAndPublish(SplitBase):
    def finals(self, r):
        final = r.rel / "linux" / "final"
        core = fake_wheel(final / "mojolearn-0.8.99-py3-none-manylinux_2_35_x86_64.whl", sev.Y)
        cuda = fake_wheel(final / "mojolearn_nvidia-0.8.99-py3-none-manylinux_2_35_x86_64.whl", data=b"cuda")
        rocm = fake_wheel(final / "mojolearn_amd-0.8.99-py3-none-manylinux_2_35_x86_64.whl", data=b"rocm")
        return core, cuda, rocm

    def receipt(self, out, core, plugins, vendor):
        """A column receipt that installed `plugins` (one wheel or several)
        beside the core; a real split column installs BOTH."""
        plugins = plugins if isinstance(plugins, (list, tuple)) else [plugins]
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(dict(
            status="PASSED", scope="expanded", source_commit=sev.Y, wheel=str(core), wheel_sha256=release.sha256(core),
            installed=dict(vendor=vendor),
            plugins=[dict(wheel="/box/" + p.name, wheel_sha256=release.sha256(p)) for p in plugins])))
        (out / f"column-{vendor}.json").write_text(json.dumps(sev.column(vendor)))
        (out / f"diff-ref-{vendor}.txt").write_text("summary: IDENTICAL=1\n")

    def test_each_column_installs_its_own_plugin(self):
        r = self.release()
        core, cuda, rocm = self.finals(r)
        ref = self.tmp / "release-check" / sev.Y[:12] / "metal"
        ref.mkdir(parents=True)
        (ref / "column.json").write_text(json.dumps(sev.column("apple-m4")))

        def selection(vendor):
            p = r.rel / f"selection-{vendor}.json"
            p.write_text(json.dumps(dict(backend=vendor, column=vendor, fixtures="base", lanes=["rf-clf"])))
            return p
        r.gpu_selection = selection
        legs = {l.name: l for l in r.column_legs()}
        nv, amd = legs["nvidia"].command, legs["amd"].command

        def plugins(cmd):
            return [cmd[i + 1] for i, a in enumerate(cmd) if a == "--plugin"]
        # BOTH plugins on every column (the core requires both), its own first
        self.assertEqual(plugins(nv), [str(cuda), str(rocm)])
        self.assertEqual(plugins(amd), [str(rocm), str(cuda)])
        self.assertEqual(legs["amd"].provenance["wheel_sha256"], release.sha256(rocm), "keyed to the plugin")
        self.assertEqual(legs["amd"].provenance["core_sha256"], release.sha256(core))
        self.assertEqual(r.column_legs(("amd",))[0].name, "amd")

    def test_a_column_is_ok_only_with_its_own_plugin_in_the_receipt(self):
        r = self.release()
        core, cuda, rocm = self.finals(r)
        self.assertFalse(r.nvidia_column_ok())
        self.receipt(r.rel / "smoke-linux", core, cuda, "cuda")         # its own plugin alone: not what pip installs
        self.assertFalse(r.nvidia_column_ok())
        self.receipt(r.rel / "smoke-linux", core, (cuda, rocm), "cuda")
        self.assertTrue(r.nvidia_column_ok())
        self.receipt(r.rel / "column-amd", core, cuda, "hip")          # the wrong plugin
        self.assertFalse(r.amd_column_ok())
        self.receipt(r.rel / "column-amd", core, rocm, "hip")          # its own alone
        self.assertFalse(r.amd_column_ok())
        other = fake_wheel(self.tmp / "other" / rocm.name, data=b"not the final")
        self.receipt(r.rel / "column-amd", core, (other, cuda), "hip")  # a plugin of other bytes
        self.assertFalse(r.amd_column_ok())
        self.receipt(r.rel / "column-amd", core, (rocm, cuda), "hip")
        self.assertTrue(r.amd_column_ok())

    def test_the_core_publishes_only_when_both_columns_passed(self):
        r = self.release()
        core, cuda, rocm = self.finals(r)
        with self.assertRaisesRegex(release.StepFailed, "not PASSED: NVIDIA, AMD"):
            r.core_receipt()
        self.receipt(r.rel / "column-amd", core, (rocm, cuda), "hip")
        with self.assertRaisesRegex(release.StepFailed, "not PASSED: NVIDIA$"):
            r.core_receipt()
        self.receipt(r.rel / "smoke-linux", core, (cuda, rocm), "cuda")
        self.assertEqual(r.core_receipt(), r.rel / "smoke-linux" / "results.json")
        (r.rel / "column-amd" / "results.json").unlink()
        with self.assertRaisesRegex(release.StepFailed, "not PASSED: AMD$"):
            r.core_receipt()

    def test_each_package_is_its_own_publication(self):
        calls = []
        r = self.release(publish="testpypi")
        r.runner = lambda cmd, env, log, detach=False: calls.append([str(c) for c in cmd]) or 0
        r.on_pypi = lambda wheel: False
        core, cuda, rocm = self.finals(r)
        self.receipt(r.rel / "smoke-linux", core, (cuda, rocm), "cuda")
        self.receipt(r.rel / "column-amd", core, (rocm, cuda), "hip")
        for step in ("publish_nvidia", "publish_amd", "publish_core_linux"):
            result, data = getattr(r, "step_" + step)()
            self.assertIn("testpypi via alpha-api-0.8.99-", result)
        published = [(pathlib.Path(c[2]).name.split("-")[0], c[3].split("-")[3], pathlib.Path(c[-1]).parent.name)
                     for c in calls]
        self.assertEqual(published, [("mojolearn_nvidia", "nvidia", "smoke-linux"), ("mojolearn_amd", "amd", "column-amd"),
                                     ("mojolearn", "linux", "smoke-linux")])

    def test_a_plugin_is_not_published_on_a_receipt_that_did_not_install_it(self):
        r = self.release(publish="testpypi")
        r.runner = lambda cmd, env, log, detach=False: 0
        r.on_pypi = lambda wheel: False
        core, cuda, rocm = self.finals(r)
        self.receipt(r.rel / "column-amd", core, cuda, "hip")
        with self.assertRaisesRegex(release.StepFailed, "did not install mojolearn_amd"):
            r.step_publish_amd()

    def test_on_pypi_asks_the_plugins_own_project(self):
        r = self.release()
        _, cuda, _ = self.finals(r)
        seen = []

        def urlopen(url, timeout=None):
            seen.append(url)
            raise OSError("offline")
        with mock.patch.object(release.urllib.request, "urlopen", urlopen):
            self.assertFalse(r.on_pypi(cuda))
        self.assertEqual(seen, ["https://pypi.org/pypi/mojolearn-nvidia/0.8.99/json"])


class Reuse(unittest.TestCase):
    def test_a_published_split_set_is_merged_back_for_the_assembly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            core = tmp / "mojolearn-1-py3-none-manylinux_2_35_x86_64.whl"
            with zipfile.ZipFile(core, "w") as z:
                z.writestr("mojolearn/__init__.py", "")
                z.writestr("mojolearn-1.dist-info/LINUX_PAYLOAD.json", json.dumps(dict(a=1, split=dict(role="core-linux"))))
            cuda = tmp / "mojolearn_nvidia-1-py3-none-manylinux_2_35_x86_64.whl"
            with zipfile.ZipFile(cuda, "w") as z:
                z.writestr("mojolearn/cuda/sm_89/_mojolearn_knn.so", "bin")
                z.writestr("mojolearn_nvidia-1.dist-info/METADATA", "x")
            merged = release.merge_split([core, cuda], tmp / "out" / core.name)
            with zipfile.ZipFile(merged) as z:
                self.assertEqual(sorted(z.namelist()), ["mojolearn-1.dist-info/LINUX_PAYLOAD.json",
                                                        "mojolearn/__init__.py", "mojolearn/cuda/sm_89/_mojolearn_knn.so"])
                self.assertEqual(json.loads(z.read("mojolearn-1.dist-info/LINUX_PAYLOAD.json")), dict(a=1))


if __name__ == "__main__":
    unittest.main()
