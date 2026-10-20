"""THE SPLIT LINUX PACKAGES IN `pixi run release` (the only Linux layout),
proved without a rental: the per-package
pipelines and their gates (2026-10-20: mojolearn-core FIRST, each vendor on
its own column after the core, the file-free Linux mojolearn LAST, after the
core and both vendors published), the pack of the five wheels through audit.sh
and the strip, the columns' --plugin, and the per-package publication. The runner stands in for every process; the pack
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
    def release(self, commit=sev.Y, **kw):
        return super().release(commit=commit, **kw)


class Switch(SplitBase):
    def test_the_split_layout_is_the_only_one(self):
        """The combined Linux wheel is gone (2026-09-26, after 0.8.22
        went out split end to end); so are --split-linux and --combined-linux."""
        r = self.release()
        self.assertEqual(r.STEPS, release.Release.STEPS)
        self.assertIn("publish-core-linux", r.STEPS)
        self.assertNotIn("publish-linux", r.STEPS)
        self.assertNotIn("gpu-columns", r.STEPS)
        self.assertFalse(hasattr(release, "SPLIT_LINUX_ENV"))
        for flag in ("--combined-linux", "--split-linux"):
            with self.assertRaises(SystemExit) as cm, mock.patch("sys.stderr"):
                release.main(["0.8.99", "--dry-run", flag, "--state-dir", str(self.tmp / "s")])
            self.assertEqual(cm.exception.code, 2)

    def test_a_release_packed_combined_is_refused_until_repacked(self):
        r = self.release()
        r.state["linux_layout"] = "combined"
        r.save()
        with self.assertRaisesRegex(SystemExit, "--redo linux-pack"):
            self.release()
        self.release(redo="linux-pack")

    def test_the_split_pipelines(self):
        r = self.release()
        self.assertEqual(list(r.PIPELINES), ["macos", "core-linux", "nvidia", "nvidia-ptx", "amd", "meta-linux"])
        for p in r.PIPELINES.values():
            for s in p["builds"] + p["checks"] + [p["publish"]]:
                self.assertIn(s, r.STEPS)
        # THE CORE FIRST (it requires no vendor), THE LINUX mojolearn LAST
        # (it requires the core and both vendors)
        # Andrew 2026-10-10: identity runs ONCE; linux-joint-diff is deleted
        self.assertEqual(set(r.NEEDS["publish-core-linux"]), {"linux-pack"})
        self.assertEqual(set(r.AFTER["publish-core-linux"]), {"gpu-column-nvidia", "gpu-column-amd"})
        self.assertEqual(set(r.NEEDS["publish-nvidia-ptx"]), {"gpu-column-nvidia", "publish-core-linux"})
        self.assertEqual(set(r.NEEDS["publish-nvidia"]), {"gpu-column-nvidia", "gpu-column-nvidia-hopper", "publish-nvidia-ptx"})
        self.assertEqual(set(r.NEEDS["publish-amd"]), {"gpu-column-amd", "publish-core-linux"})
        self.assertEqual(set(r.NEEDS["publish-meta-linux"]), {"publish-core-linux", "publish-nvidia", "publish-amd"})
        self.assertEqual(r.PIPELINES["meta-linux"]["platform"], "linux-meta")
        self.assertNotIn("linux-joint-diff", r.AFTER)

    def test_the_split_packages_and_what_pip_resolves(self):
        self.assertEqual(release.SPLIT_PACKAGES, {
            "linux": ("mojolearn-core", "mojolearn_core"), "nvidia": ("mojolearn-nvidia", "mojolearn_nvidia"),
            "amd": ("mojolearn-amd", "mojolearn_amd"), "nvidia-ptx": ("mojolearn-nvidia-ptx", "mojolearn_nvidia_ptx"),
            "linux-meta": ("mojolearn", "mojolearn")})
        self.assertEqual(set(release.PLUGIN_PACKAGES), {"nvidia", "amd", "nvidia-ptx"})
        self.assertEqual(release.LINUX_RESOLVES["linux"], ("linux",))
        self.assertEqual(set(release.LINUX_RESOLVES["linux-meta"]), set(release.SPLIT_PACKAGES))
        self.assertEqual(set(release.LINUX_RESOLVES["nvidia"]), {"linux", "nvidia", "nvidia-ptx"})
        self.assertEqual(set(release.LINUX_RESOLVES["amd"]), {"linux", "amd"})
        self.assertNotIn("nvidia-ptx", release.LINUX_RESOLVES)


class Gates(SplitBase):
    """Every step stood in for; which packages publish when a column or the
    core's publish fails."""

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

    def test_everything_passing_publishes_the_five_packages_and_macos(self):
        r, order = self.staged()
        self.assertEqual(r.go(), 0)
        self.assertEqual(self.published(r), ["amd", "linux", "linux-meta", "macos", "nvidia", "nvidia-ptx"])
        # the core first: each vendor requires mojolearn-core at <v>
        self.assertLess(order.index("publish-core-linux"), order.index("publish-nvidia-ptx"))
        self.assertLess(order.index("publish-core-linux"), order.index("publish-amd"))
        self.assertLess(order.index("publish-nvidia-ptx"), order.index("publish-nvidia"))
        # the Linux mojolearn last: pip resolves mojolearn==<v> only once the
        # core and both vendors at <v> are on the index
        for step in ("publish-core-linux", "publish-nvidia", "publish-amd"):
            self.assertGreater(order.index("publish-meta-linux"), order.index(step))

    def test_a_failed_amd_column_holds_amd_and_the_linux_mojolearn(self):
        r, order = self.staged(fail={"gpu-column-amd"})
        self.assertEqual(r.go(), 1)
        # the core requires no vendor: it and NVIDIA still publish
        self.assertEqual(self.published(r), ["linux", "macos", "nvidia", "nvidia-ptx"])
        self.assertNotIn("publish-amd", order)
        self.assertNotIn("publish-meta-linux", order)
        self.assertIn("amd: FAILED at gpu-column-amd", "\n".join(r.lines))

    def test_a_failed_nvidia_column_holds_nvidia_and_the_linux_mojolearn(self):
        r, order = self.staged(fail={"gpu-column-nvidia"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "linux", "macos"])
        for step in ("publish-nvidia-ptx", "publish-nvidia", "publish-meta-linux"):
            self.assertNotIn(step, order)

    def test_failed_hopper_holds_its_vendor_and_the_linux_mojolearn(self):
        r, order = self.staged(fail={"gpu-column-nvidia-hopper"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "linux", "macos", "nvidia-ptx"])
        for step in ("publish-nvidia", "publish-meta-linux"):
            self.assertNotIn(step, order)

    def test_a_failed_nvidia_publish_holds_the_linux_mojolearn(self):
        r, order = self.staged(fail={"publish-nvidia"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "linux", "macos", "nvidia-ptx"])
        self.assertNotIn("publish-meta-linux", order)

    def test_a_failed_amd_publish_holds_the_linux_mojolearn(self):
        r, order = self.staged(fail={"publish-amd"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["linux", "macos", "nvidia", "nvidia-ptx"])
        self.assertNotIn("publish-meta-linux", order)

    def test_a_failed_core_publish_holds_every_linux_package(self):
        r, order = self.staged(fail={"publish-core-linux"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["macos"])
        for step in ("publish-nvidia-ptx", "publish-nvidia", "publish-amd", "publish-meta-linux"):
            self.assertNotIn(step, order)

    def test_failed_ptx_publish_holds_nvidia_and_the_linux_mojolearn(self):
        r, order = self.staged(fail={"publish-nvidia-ptx"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "linux", "macos"])
        self.assertNotIn("publish-nvidia", order)
        self.assertNotIn("publish-meta-linux", order)

    def test_a_failed_linux_mojolearn_publish_leaves_the_rest_published(self):
        r, order = self.staged(fail={"publish-meta-linux"})
        self.assertEqual(r.go(), 1)
        self.assertEqual(self.published(r), ["amd", "linux", "macos", "nvidia", "nvidia-ptx"])

class Pack(SplitBase):
    """linux-pack: pack (the real packer over fake sets), audit
    each wheel core first, strip each, split_audit the final set."""

    def test_five_wheels_packed_audited_stripped_and_audited_as_a_set(self):
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
        self.assertEqual(audits, ["mojolearn_core", "mojolearn_nvidia", "mojolearn_amd", "mojolearn_nvidia_ptx",
                                  "mojolearn"], "the core first, the file-free Linux mojolearn last")
        self.assertEqual(sum(1 for c in calls if c[1].endswith("strip_wheel_dir_entries.py")), 5)
        self.assertTrue(finals["linux"].name.startswith("mojolearn_core-"), finals["linux"])
        self.assertTrue(finals["linux-meta"].name.startswith("mojolearn-"), finals["linux-meta"])
        report = json.loads((r.rel / "linux" / "final" / "split-audit.json").read_text())
        self.assertEqual(report["problems"], [])
        self.assertEqual(r.state["linux_layout"], "split")
        self.assertIn("mojolearn_amd-", result)
        # A rerun takes the six finals as they are
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
        core = fake_wheel(final / "mojolearn_core-0.8.99-py3-none-manylinux_2_35_x86_64.whl", sev.Y)
        self.meta = fake_wheel(final / "mojolearn-0.8.99-py3-none-manylinux_2_35_x86_64.whl", data=b"meta")
        cuda = fake_wheel(final / "mojolearn_nvidia-0.8.99-py3-none-manylinux_2_35_x86_64.whl", data=b"cuda")
        rocm = fake_wheel(final / "mojolearn_amd-0.8.99-py3-none-manylinux_2_35_x86_64.whl", data=b"rocm")
        ptx = fake_wheel(final / "mojolearn_nvidia_ptx-0.8.99-py3-none-manylinux_2_35_x86_64.whl", data=b"ptx")
        self.plugins = [cuda, rocm, ptx]
        return core, cuda, rocm

    def receipt(self, out, core, plugins, vendor, arch=None):
        """A column receipt that installed `plugins` (one wheel or several)
        beside the core; a real split column installs all five GPU distributions."""
        plugins = plugins if isinstance(plugins, (list, tuple)) else [plugins]
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(dict(
            status="PASSED", scope="expanded", source_commit=sev.Y, wheel=str(core), wheel_sha256=release.sha256(core),
            installed=dict(vendor=vendor, gpu_arch=arch or ("sm_89" if vendor == "cuda" else "gfx942")),
            plugins=[dict(wheel="/box/" + p.name, wheel_sha256=release.sha256(p)) for p in plugins])))

    def test_each_column_installs_its_own_plugin(self):
        r = self.release()
        core, cuda, rocm = self.finals(r)
        legs = {l.name: l for l in r.column_legs()}
        nv, amd = legs["nvidia"].command, legs["amd"].command

        def plugins(cmd):
            return [cmd[i + 1] for i, a in enumerate(cmd) if a == "--plugin"]
        # All native payloads and both aggregates on every column, its aggregate first
        self.assertEqual(plugins(nv), [str(p) for p in self.plugins])
        self.assertEqual(plugins(amd), [str(rocm), *[str(p) for p in self.plugins if p != rocm]])
        self.assertEqual(plugins(legs["nvidia-hopper"].command), [str(p) for p in self.plugins])
        self.assertEqual(legs["amd"].provenance["wheel_sha256"], release.sha256(r.split_final("amd")), "keyed to the native payload")
        self.assertEqual(legs["amd"].provenance["core_sha256"], release.sha256(core))
        self.assertEqual(r.column_legs(("amd",))[0].name, "amd")

    def test_a_column_is_ok_only_with_its_own_plugin_in_the_receipt(self):
        r = self.release()
        core, cuda, rocm = self.finals(r)
        self.assertFalse(r.nvidia_column_ok())
        self.receipt(r.rel / "smoke-linux", core, cuda, "cuda")         # its own plugin alone: not what pip installs
        self.assertFalse(r.nvidia_column_ok())
        self.receipt(r.rel / "smoke-linux", core, self.plugins, "cuda")
        self.assertTrue(r.nvidia_column_ok())
        self.receipt(r.rel / "column-amd", core, cuda, "hip")          # the wrong plugin
        self.assertFalse(r.amd_column_ok())
        self.receipt(r.rel / "column-amd", core, rocm, "hip")          # its own alone
        self.assertFalse(r.amd_column_ok())
        other = fake_wheel(self.tmp / "other" / rocm.name, data=b"not the final")
        self.receipt(r.rel / "column-amd", core, [other, *[p for p in self.plugins if p != rocm]], "hip")  # a plugin of other bytes
        self.assertFalse(r.amd_column_ok())
        self.receipt(r.rel / "column-amd", core, self.plugins, "hip")
        self.assertTrue(r.amd_column_ok())

    def test_complete_wheel_receipt_cannot_qualify_the_wrong_architecture(self):
        r = self.release()
        core, _, _ = self.finals(r)
        self.receipt(r.rel / "column-nvidia-hopper", core, self.plugins, "cuda", "sm_89")
        self.assertFalse(r.hopper_column_ok(), "Ada execution does not qualify Hopper payload")
        self.receipt(r.rel / "column-nvidia-hopper", core, self.plugins, "cuda", "sm_90a")
        self.assertTrue(r.hopper_column_ok())
        self.receipt(r.rel / "smoke-linux", core, self.plugins, "cuda", "sm_90a")
        self.assertFalse(r.nvidia_column_ok(), "Hopper execution does not qualify Ada payload")

    def test_the_core_publishes_on_either_passed_vendor_column(self):
        """mojolearn-core requires no vendor (2026-10-20): one PASSED column
        that installed it is enough, NVIDIA's first, else AMD's."""
        r = self.release()
        core, cuda, rocm = self.finals(r)
        with self.assertRaisesRegex(release.StepFailed, "neither NVIDIA nor AMD PASSED"):
            r.core_receipt()
        self.receipt(r.rel / "column-amd", core, self.plugins, "hip")
        self.assertEqual(r.core_receipt(), r.rel / "column-amd" / "results.json")
        self.receipt(r.rel / "smoke-linux", core, self.plugins, "cuda")
        self.assertEqual(r.core_receipt(), r.rel / "smoke-linux" / "results.json")
        (r.rel / "column-amd" / "results.json").unlink()
        self.assertEqual(r.core_receipt(), r.rel / "smoke-linux" / "results.json", "no Hopper or AMD needed")

    def test_each_package_is_its_own_publication(self):
        calls = []
        r = self.release(publish="testpypi")
        r.runner = lambda cmd, env, log, detach=False: calls.append([str(c) for c in cmd]) or 0
        r.on_pypi = lambda wheel: False
        core, cuda, rocm = self.finals(r)
        self.receipt(r.rel / "smoke-linux", core, self.plugins, "cuda")
        self.receipt(r.rel / "column-amd", core, self.plugins, "hip")
        self.receipt(r.rel / "column-nvidia-hopper", core, self.plugins, "cuda", "sm_90a")
        for step in ("publish_core_linux", "publish_nvidia", "publish_amd", "publish_meta_linux"):
            result, data = getattr(r, "step_" + step)()
            self.assertIn("testpypi via alpha-api-0.8.99-", result)
        # (package, the receipt it publishes on, or its route when it has none)
        published = [(pathlib.Path(c[2]).name.split("-")[0],
                      pathlib.Path(c[-1]).parent.name if c[-2] == "--light-smoke" else c[-1]) for c in calls]
        self.assertEqual(published, [("mojolearn_core", "smoke-linux"), ("mojolearn_nvidia", "smoke-linux"),
                                     ("mojolearn_amd", "column-amd"), ("mojolearn", "--preverified")])
        self.assertIn("alpha-api-0.8.99-linux-meta-", calls[-1][3])

    def test_finish_resolves_each_linux_install_to_its_package_set(self):
        """pip install mojolearn installs everything; mojolearn-nvidia the core,
        NVIDIA and its PTX package; mojolearn-amd the core and AMD; each at the
        published bytes (Release.finish_resolve, 2026-10-20)."""
        r = self.release(publish="pypi")
        self.finals(r)
        finals = r.split_finals()
        by_project = {project: k for k, (project, _) in release.SPLIT_PACKAGES.items()}
        served = {}

        def runner(cmd, env, log, detach=False):
            cmd = [str(c) for c in cmd]
            dest = pathlib.Path(cmd[cmd.index("-d") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            project = cmd[-1].split("==")[0]
            for k in served.get(project, release.LINUX_RESOLVES[by_project[project]]):
                shutil.copy2(finals[k], dest / finals[k].name)
            return 0
        r.runner = runner
        venv = self.tmp / "venv"
        self.assertIn("resolves mojolearn-core, mojolearn-nvidia, mojolearn-nvidia-ptx", r.finish_resolve(venv, "nvidia"))
        self.assertIn("resolves mojolearn-core, mojolearn-amd", r.finish_resolve(venv, "amd"))
        self.assertIn("resolves mojolearn-core, mojolearn-nvidia, mojolearn-amd, mojolearn-nvidia-ptx, mojolearn",
                      r.finish_resolve(venv, "linux-meta"))
        # a resolution without a vendor, or with other bytes, is refused
        served["mojolearn"] = ("linux-meta", "linux", "nvidia", "nvidia-ptx")
        with self.assertRaisesRegex(release.StepFailed, "does not resolve exactly"):
            r.finish_resolve(venv, "linux-meta")
        served.clear()
        finals["amd"] = fake_wheel(self.tmp / "other" / finals["amd"].name, data=b"other bytes")
        with self.assertRaisesRegex(release.StepFailed, "mojolearn-amd==0.8.99 .* does not resolve exactly"):
            r.finish_resolve(venv, "amd")

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
    def test_a_published_mojolearn_core_set_is_merged_back_for_the_assembly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            core = tmp / "mojolearn_core-1-py3-none-manylinux_2_35_x86_64.whl"
            with zipfile.ZipFile(core, "w") as z:
                z.writestr("mojolearn/__init__.py", "")
                z.writestr("mojolearn_core-1.dist-info/LINUX_PAYLOAD.json",
                           json.dumps(dict(a=1, split=dict(role="core-linux"))))
            cuda = tmp / "mojolearn_nvidia-1-py3-none-manylinux_2_35_x86_64.whl"
            with zipfile.ZipFile(cuda, "w") as z:
                z.writestr("mojolearn/cuda/sm_89/_mojolearn_knn.so", "bin")
                z.writestr("mojolearn_nvidia-1.dist-info/METADATA", "x")
            # the plugin listed first: the core is found by its name, not its place
            merged = release.merge_split([cuda, core], tmp / "out" / core.name)
            with zipfile.ZipFile(merged) as z:
                self.assertEqual(sorted(z.namelist()), ["mojolearn/__init__.py", "mojolearn/cuda/sm_89/_mojolearn_knn.so",
                                                        "mojolearn_core-1.dist-info/LINUX_PAYLOAD.json"])
                self.assertEqual(json.loads(z.read("mojolearn_core-1.dist-info/LINUX_PAYLOAD.json")), dict(a=1))

    def test_a_published_split_set_is_merged_back_for_the_assembly(self):
        """A release before 2026-10-20: the core was mojolearn-* with files."""
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
