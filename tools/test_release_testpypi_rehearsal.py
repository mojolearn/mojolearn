"""THE TESTPYPI REHEARSAL (Andrew 2026-11-06): inside --publish pypi, testpypi-rehearsal uploads the gated wheels
to TestPyPI, resolves every package set there with our names from TestPyPI only, compares every sha256 and installs
and imports the macOS wheel, all BEFORE the first real publish; after the publish only hashes are compared.
Pure Python: fake TestPyPI, fake pip, fake publisher; nothing reaches a network index, a box or a device."""
import json
import pathlib
import sys
import unittest
import urllib.request
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import release  # noqa: E402
import release_testpypi_rehearsal as TR  # noqa: E402
import test_release_severable as sev  # noqa: E402

V = "0.8.99"
LINUX = "manylinux_2_35_x86_64"
MAC = "macosx_11_0_arm64"
PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")
EXTRAS = ("numpy", "verify", "torch", "max")


def wheel(path, name, extras=(), data=b"x", commit=None):
    """A wheel with real METADATA (name, version, Requires-Python, the Python classifiers, its extras)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {V}", "Requires-Python: >=3.10"]
    meta += [f"Classifier: Programming Language :: Python :: {p}" for p in PYTHONS]
    meta += [f"Provides-Extra: {e}" for e in extras]
    dist = path.name.split("-")[0] + f"-{V}.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{dist}/METADATA", "\n".join(meta) + "\n")
        if commit:
            z.writestr("mojolearn/identity_columns/COMMIT", commit + "\n")
        z.writestr("payload", data)
    return path


class FakeIndex:
    """RoutingIndex's interface: what the judge needs (served URLs per side) and nothing on a network."""
    url = "http://127.0.0.1:9/simple/"

    def __init__(self):
        self.served = dict(testpypi=set(), pypi=set())
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def served_url(name):
    return f"https://test-files.pythonhosted.org/packages/aa/{name}"


class Rehearsal(sev.Base):
    """A release frozen at Y with every check before the publishes done, the five gated wheels on disk."""

    def setUp(self):
        super().setUp()
        self.uploads, self.commands, self.on_testpypi = [], [], {}
        self.served_sha = {}          # file name -> sha256 TestPyPI advertises (default: the gated one)
        self.served_bytes = {}        # file name -> bytes pip download returns (default: the gated file)
        self.resolved = {}            # root project -> the files it resolves to (default: the expected set)
        self.import_rc = 0
        self.index = FakeIndex()
        patches = [mock.patch.object(TR, "RoutingIndex", lambda: self.index),
                   mock.patch.object(TR, "testpypi_file_state", self.file_state)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def make(self, publish="pypi"):
        r = self.release(publish=publish)
        final = r.rel / "linux" / "final"
        self.w = dict(
            linux=wheel(final / f"mojolearn_core-{V}-py3-none-{LINUX}.whl", "mojolearn-core", EXTRAS, b"core", sev.Y),
            nvidia=wheel(final / f"mojolearn_nvidia-{V}-py3-none-{LINUX}.whl", "mojolearn-nvidia", (), b"cuda"),
            amd=wheel(final / f"mojolearn_amd-{V}-py3-none-{LINUX}.whl", "mojolearn-amd", (), b"rocm"),
            **{"linux-meta": wheel(final / f"mojolearn-{V}-py3-none-{LINUX}.whl", "mojolearn", EXTRAS, b"meta")},
            macos=wheel(r.rel / "macos" / f"mojolearn-{V}-py3-none-{MAC}.whl", "mojolearn", EXTRAS, b"mac", sev.Y))
        for step in r.STEPS:
            if step not in release.PUBLISH_STEPS and step != release.REHEARSAL_STEP:
                r.state["steps"][step] = dict(done=True, commit=sev.Y, at="t")
        r.smoke_python = lambda: "python3.12"
        for platform, step in r.platform_publish().items():
            setattr(r, "step_" + step.replace("-", "_"), self.publisher(r, platform))
        r.runner = self.runner
        return r

    # the fakes
    def file_state(self, project, version, filename, sha, fetch=None):
        served = self.on_testpypi.get(filename)
        if served is None:
            return "absent", None
        return ("same" if served == sha else "different"), served

    def publisher(self, r, platform):
        def publish():
            # the rehearsal reaches the publisher with TestPyPI as the target, never the real PyPI
            self.assertEqual(getattr(r._tl, "publish_target", None), "testpypi")
            self.uploads.append(platform)
            self.on_testpypi[self.w[platform].name] = release.sha256(self.w[platform])
            return "testpypi via alpha-api-tag", {}
        return publish

    def by_project(self):
        out = {}
        for k, w in self.w.items():
            out.setdefault(TR.wheel_project(w), []).append(w)
        return out

    def runner(self, cmd, env, log, detach=False):
        cmd = [str(c) for c in cmd]
        self.commands.append(cmd)
        if "--dry-run" in cmd:
            root = cmd[-1].split("==")[0].split("[")[0]
            plat = cmd[cmd.index("--platform") + 1]
            pick = (lambda w: w.name.endswith(MAC + ".whl")) if plat == MAC else (lambda w: w.name.endswith(LINUX + ".whl"))
            members = self.resolved.get(root)
            if members is None:
                key = "macos" if plat == MAC else next(k for k, (p, _) in release.SPLIT_PACKAGES.items() if p == root)
                members = [self.w[m] for m in (("macos",) if key == "macos" else release.LINUX_RESOLVES[key])]
            install = []
            links = pathlib.Path(cmd[cmd.index("--find-links") + 1]) if "--find-links" in cmd else None
            for w in members:
                if links and (links / w.name).is_file():
                    url = (links / w.name).as_uri()
                else:
                    url = served_url(w.name)
                    self.index.served["testpypi"].add(url)
                install.append(dict(metadata=dict(name=TR.wheel_project(w), version=V), download_info=dict(
                    url=url, archive_info=dict(hashes=dict(sha256=self.served_sha.get(w.name, release.sha256(w)))))))
            pathlib.Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(dict(install=install)))
            return 0
        if "download" in cmd:
            dest = pathlib.Path(cmd[cmd.index("--dest") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            project = cmd[-1].split("==")[0]
            plat = cmd[cmd.index("--platform") + 1]
            w = next(w for w in self.by_project()[project] if w.name.endswith(plat + ".whl")
                     or (plat.startswith("macosx") and "macosx" in w.name))
            (dest / w.name).write_bytes(self.served_bytes.get(w.name, w.read_bytes()))
            return 0
        if cmd[-2:-1] == ["-c"] or "-I" in cmd:
            return self.import_rc
        return 0

    def rehearse(self, r):
        return r.run_step(release.REHEARSAL_STEP)


class Order(sev.Base):
    staged = sev.Pipelines.staged   # every step stood in for (test_release_severable.py)

    def test_the_rehearsal_runs_before_any_publish_step(self):
        for publish in release.PUBLISH_STEPS:
            self.assertIn(release.REHEARSAL_STEP, release.NEEDS[publish])
        r, order = self.staged()
        self.assertEqual(r.go(), 0)
        first = min(order.index(p) for p in release.PUBLISH_STEPS)
        self.assertLess(order.index(release.REHEARSAL_STEP), first)
        for gate in ("gate-nvidia", "gate-amd", "gate-apple", "macos-smoke", "linux-pack"):
            self.assertLess(order.index(gate), order.index(release.REHEARSAL_STEP))

    def test_a_failed_rehearsal_publishes_nothing(self):
        r, order = self.staged(fail={release.REHEARSAL_STEP})
        self.assertEqual(r.go(), 1)
        for publish in release.PUBLISH_STEPS:
            self.assertNotIn(publish, order)
            self.assertFalse(r.recorded(publish))

    def test_a_failed_mac_pipeline_still_rehearses_and_publishes_linux(self):
        r, order = self.staged(fail={"macos-smoke"})
        self.assertEqual(r.go(), 1)
        self.assertIn(release.REHEARSAL_STEP, order)
        self.assertTrue(r.recorded("publish-core-linux"))


class Steps(Rehearsal):
    def test_a_clean_rehearsal_uploads_in_order_and_records_every_sha(self):
        r = self.make()
        self.assertEqual(self.rehearse(r), "done")
        self.assertEqual(self.uploads, ["linux", "nvidia", "amd", "linux-meta", "macos"])
        rec = r.recorded(release.REHEARSAL_STEP)
        self.assertEqual(rec["wheels"], {k: release.sha256(w) for k, w in self.w.items()})
        resolves = [c for c in self.commands if "--dry-run" in c]
        roots = {c[-1] for c in resolves}
        for root in (f"mojolearn-core=={V}", f"mojolearn-nvidia=={V}", f"mojolearn-amd=={V}", f"mojolearn=={V}",
                     f"mojolearn[torch]=={V}", f"mojolearn-core[max]=={V}"):
            self.assertIn(root, roots)
        self.assertEqual({c[c.index("--python-version") + 1] for c in resolves}, set(PYTHONS))
        self.assertEqual({c[c.index("--platform") + 1] for c in resolves}, {LINUX, MAC})
        # the macOS wheel installs pinned to the gated sha256 and imports
        install = next(c for c in self.commands if "install" in c and "--require-hashes" in c)
        req = pathlib.Path(install[install.index("-r") + 1]).read_text()
        self.assertIn(release.sha256(self.w["macos"]), req)
        self.assertTrue(any("import mojolearn" in " ".join(c) for c in self.commands))

    def test_exists_with_the_same_bytes_passes_without_an_upload(self):
        r = self.make()
        self.on_testpypi = {w.name: release.sha256(w) for w in self.w.values()}
        self.assertEqual(self.rehearse(r), "done")
        self.assertEqual(self.uploads, [])

    def test_a_resumed_release_does_not_rehearse_again(self):
        r = self.make()
        self.assertEqual(self.rehearse(r), "done")
        self.uploads.clear(), self.commands.clear()
        self.assertEqual(self.rehearse(r), "done")
        self.assertEqual((self.uploads, self.commands), ([], []))

    def test_exists_with_different_bytes_stops_before_publish(self):
        r = self.make()
        self.on_testpypi = {self.w["amd"].name: "f" * 64}
        self.assertEqual(self.rehearse(r), "failed")
        error = r.state["failures"][release.REHEARSAL_STEP]["error"]
        self.assertIn("never takes a file name twice", error)
        self.assertIn("next version", error)
        self.assertNotIn("amd", self.uploads)
        self.assertNoPublish(r)

    def test_a_resolution_mismatch_stops_before_publish(self):
        r = self.make()
        self.resolved["mojolearn"] = [self.w["linux-meta"], self.w["linux"], self.w["nvidia"]]  # no AMD
        self.assertEqual(self.rehearse(r), "failed")
        self.assertIn("want exactly", r.state["failures"][release.REHEARSAL_STEP]["error"])
        self.assertNoPublish(r)

    def test_a_hash_mismatch_on_the_index_stops_before_publish(self):
        r = self.make()
        self.served_sha[self.w["nvidia"].name] = "0" * 64
        self.assertEqual(self.rehearse(r), "failed")
        self.assertIn("index sha256 " + "0" * 64, r.state["failures"][release.REHEARSAL_STEP]["error"])
        self.assertNoPublish(r)

    def test_downloaded_bytes_that_differ_stop_before_publish(self):
        r = self.make()
        self.served_bytes[self.w["linux"].name] = b"other bytes"
        self.assertEqual(self.rehearse(r), "failed")
        self.assertIn("TestPyPI bytes of " + self.w["linux"].name, r.state["failures"][release.REHEARSAL_STEP]["error"])
        self.assertNoPublish(r)

    def test_an_import_failure_stops_before_publish(self):
        r = self.make()
        self.import_rc = 1
        self.assertEqual(self.rehearse(r), "failed")
        self.assertIn("import mojolearn exited 1", r.state["failures"][release.REHEARSAL_STEP]["error"])
        self.assertNoPublish(r)

    def test_a_file_over_the_testpypi_limit_is_owed_never_silent(self):
        r = self.make()
        with mock.patch.dict(TR.TESTPYPI_FILE_BYTES, {"mojolearn-nvidia": 10}):
            self.assertEqual(self.rehearse(r), "done")
        rec = r.recorded(release.REHEARSAL_STEP)
        # the NVIDIA wheel and the Linux mojolearn (which requires it on the index) are not uploaded
        self.assertEqual(self.uploads, ["linux", "amd", "macos"])
        self.assertEqual(rec["local"], ["nvidia", "linux-meta"])
        self.assertIn("OWED", rec["result"])
        self.assertIn("over TestPyPI's 10-byte limit for mojolearn-nvidia", rec["result"])
        # every check that needs no upload still ran: the sets offer the gated files locally
        self.assertTrue(any("--find-links" in c for c in self.commands if "--dry-run" in c))

    def test_publish_to_pypi_is_refused_without_a_rehearsal_of_these_bytes(self):
        r = self.make()
        r.on_pypi = lambda w: False
        with self.assertRaisesRegex(release.StepFailed, "TestPyPI rehearsal has not passed"):
            release.Release.publish(r, "macos", self.w["macos"], None)
        r.state["steps"][release.REHEARSAL_STEP] = dict(done=True, commit=sev.Y, at="t",
                                                         wheels={"macos": "e" * 64})
        with self.assertRaisesRegex(release.StepFailed, "TestPyPI rehearsal has not passed"):
            release.Release.publish(r, "macos", self.w["macos"], None)

    def test_other_publish_targets_skip_the_rehearsal(self):
        for target in ("testpypi", "none"):
            r = self.make(publish=target)
            self.assertEqual(self.rehearse(r), "done")
            self.assertIn("skipped", r.recorded(release.REHEARSAL_STEP)["result"])
        self.assertEqual(self.uploads, [])

    def assertNoPublish(self, r):  # noqa: N802
        for publish in release.PUBLISH_STEPS:
            self.assertFalse(r.recorded(publish))


class AfterPublish(Rehearsal):
    def test_the_finish_line_installs_and_runs_nothing(self):
        r = self.make()
        for step in release.PUBLISH_STEPS:
            r.state["steps"][step] = dict(done=True, commit=sev.Y, at="t", result="pypi")
        result, data = r.step_finish_line()
        self.assertEqual(sorted(data["platforms"]), sorted(self.w))
        self.assertEqual(len(self.commands), len(self.w))
        for cmd in self.commands:
            self.assertIn("download", cmd)
            self.assertIn("--no-deps", cmd)
            self.assertNotIn("install", cmd)
            self.assertNotIn("venv", cmd)
            self.assertNotIn("-c", cmd)
            self.assertEqual(cmd[cmd.index("--index-url") + 1], TR.PYPI_SIMPLE)
        self.assertIn("with the shipped sha256", result)

    def test_the_finish_line_fails_on_other_bytes(self):
        r = self.make()
        r.state["steps"]["publish-amd"] = dict(done=True, commit=sev.Y, at="t", result="pypi")
        r.state["steps"]["publish-core-linux"] = dict(done=True, commit=sev.Y, at="t", result="pypi")
        self.served_bytes[self.w["amd"].name] = b"other"
        with self.assertRaisesRegex(release.StepFailed, "mojolearn-amd==0.8.99 .* shipped sha256"):
            r.step_finish_line()

    def test_no_post_publish_step_installs_anything(self):
        """Every step after the first publish: the published-* bytes checks, finish-line, boxes-down, record."""
        post = [s for s in release.Release.STEPS if s.startswith("published-") or s in ("finish-line", "record")]
        self.assertTrue(post)
        self.assertFalse(hasattr(release.Release, "finish_macos"))
        self.assertFalse(hasattr(release.Release, "finish_resolve"))
        import inspect
        for name in ("step_finish_line", "finish_bytes", "step_published"):
            src = inspect.getsource(getattr(release.Release, name))
            self.assertNotIn('"install"', src)
            self.assertNotIn('"venv"', src)


class Routing(unittest.TestCase):
    """Our project names come from TestPyPI only, whatever pip asks for."""

    def test_route_is_by_name_and_ours_never_reach_pypi(self):
        ix = TR.RoutingIndex(page=lambda url: (404, None))
        for name in ("mojolearn", "mojolearn-core", "MojoLearn_Core", "mojolearn.nvidia", "mojolearn-amd",
                     "mojolearn-nvidia-ptx", "mojolearn-anything-new"):
            self.assertEqual(ix.route(name), "testpypi", name)
        for name in ("numpy", "torch", "max", "onnx", "mojolearnx", "notmojolearn"):
            self.assertEqual(ix.route(name), "pypi", name)

    def test_the_http_router_fetches_ours_from_testpypi_and_others_from_pypi(self):
        asked = []

        def page(url):
            asked.append(url)
            return 200, dict(meta={"api-version": "1.0"}, name="x",
                             files=[dict(filename="f.whl", url="../../packages/f.whl", hashes={"sha256": "a"})])
        with TR.RoutingIndex(testpypi="https://test.example/simple/", pypi="https://real.example/simple/",
                             page=page) as ix:
            for name in ("Mojolearn_Core", "numpy"):
                with urllib.request.urlopen(ix.url + name + "/") as r:
                    doc = json.load(r)
                    self.assertEqual(r.headers["Content-Type"], TR.SIMPLE_JSON)
                self.assertTrue(doc["files"][0]["url"].startswith("https://"))
        self.assertEqual(asked, ["https://test.example/simple/mojolearn-core/", "https://real.example/simple/numpy/"])
        self.assertEqual(ix.served["testpypi"], {"https://test.example/packages/f.whl"})
        self.assertEqual(ix.served["pypi"], {"https://real.example/packages/f.whl"})

    def test_pip_runs_isolated_with_the_router_as_its_only_index(self):
        for cmd in (TR.resolve_command("py", "http://r/simple/", "mojolearn==1", LINUX, "3.12", "rep.json"),
                    TR.download_command("py", "http://r/simple/", "mojolearn", "1", LINUX, "d"),
                    TR.install_command(pathlib.Path("v"), "http://r/simple/", "req.txt")):
            self.assertIn("--isolated", cmd)
            self.assertEqual([c for c in cmd if "index" in c], ["--index-url"])
            self.assertEqual(cmd[cmd.index("--index-url") + 1], "http://r/simple/")

    def test_the_judge_refuses_our_name_from_anywhere_but_testpypi(self):
        good = "https://test-files.pythonhosted.org/packages/x/mojolearn_amd-1-py3-none-any.whl"
        bad = "https://files.pythonhosted.org/packages/x/mojolearn_amd-1-py3-none-any.whl"
        expected = {"mojolearn-amd": ("mojolearn_amd-1-py3-none-any.whl", "s")}
        served = dict(testpypi={good}, pypi={bad})

        def report(url):
            return dict(install=[dict(metadata=dict(name="mojolearn-amd", version="1"),
                                      download_info=dict(url=url, archive_info=dict(hashes=dict(sha256="s"))))])
        self.assertEqual(TR.judge_resolution(report(good), expected, "1", served), [])
        problems = TR.judge_resolution(report(bad), expected, "1", served)
        self.assertTrue(any("not from a TestPyPI file the router served" in p for p in problems), problems)

    def test_the_judge_refuses_a_third_party_project_from_testpypi(self):
        url = "https://test-files.pythonhosted.org/packages/x/numpy-9-py3-none-any.whl"
        report = dict(install=[dict(metadata=dict(name="numpy", version="9"),
                                    download_info=dict(url=url, archive_info=dict(hashes=dict(sha256="s"))))])
        problems = TR.judge_resolution(report, {}, "1", dict(testpypi={url}, pypi=set()))
        self.assertTrue(any("third-party projects come from PyPI only" in p for p in problems), problems)

    def test_file_state_reads_the_testpypi_json_api(self):
        doc = dict(urls=[dict(filename="a.whl", digests=dict(sha256="s"))])
        fetch = lambda url: (200, doc)  # noqa: E731
        self.assertEqual(TR.testpypi_file_state("mojolearn", "1", "a.whl", "s", fetch), ("same", "s"))
        self.assertEqual(TR.testpypi_file_state("mojolearn", "1", "a.whl", "t", fetch), ("different", "s"))
        self.assertEqual(TR.testpypi_file_state("mojolearn", "1", "b.whl", "s", fetch), ("absent", None))
        self.assertEqual(TR.testpypi_file_state("mojolearn", "1", "a.whl", "s", lambda u: (404, None)),
                         ("absent", None))


if __name__ == "__main__":
    unittest.main()
